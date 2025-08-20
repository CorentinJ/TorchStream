import logging
from typing import List, Tuple

from z3 import And, Bool, Implies, Int, Ints, Not, Or, Solver, sat

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_stream_params import get_streaming_params
from torchstream.sliding_window.threshold_harvester import ThresholdHarvester

logger = logging.getLogger(__name__)

_MAX_COST_FLAT_LIMIT = 100_000
_MAX_COST_REL_LIMIT = 3.0


class SlidingWindowParamsSampler:
    """
    TODO: doc
    TODO: sort these out
    - The input size to the number of windows is a deterministic, stepwise monotonic linear function.

    TODO: padding
    - The most common padding is constant, with zeroes. But there are other padding methods such as reflect or
    circular that sample from the input array to determine the padding values. How does this compromise results from
    the nan trick? Does the transform still qualify as a sliding window approach (e.g. if it reads in the middle of
    the vector to get padding values)?
    """

    def __init__(self):
        # NOTE: possible constructor parameters:
        # - Left padding variable

        # Define the parameters we're trying to uniquely determine
        self.optimizer = Solver()
        # k_i and s_i are respectively the input kernel size and stride (NOTE: it's technically the kernel span,
        # i.e. the whole span of the kernel if dilation > 1)
        self.k_i, self.s_i = Ints("k_i s_i")
        # It would be highly unusual to have a stride larger than the kernel size, leading to inputs being dropped.
        self.optimizer.add(self.k_i >= self.s_i, self.s_i > 0)
        # k_o and s_o are respectively the output kernel size and stride. These are both 1 for normal convolutions,
        # but not for transposed convolutions.
        self.k_o, self.s_o = Ints("k_o s_o")
        # Again, it would be strange to have a stride larger than the kernel size, leading to gaps in the output.
        self.optimizer.add(self.k_o >= self.s_o, self.s_o > 0)
        # The left input padding. I have not yet seen a case where varying the left padding is useful, so we'll
        # assume it constant. Also, there is no point in making the padding higher than the kernel size, as it would
        # waste compute on constant values.
        self.p_l = Int("p_l")
        self.optimizer.add(0 <= self.p_l, self.p_l < self.k_i)
        # TODO doc
        self.p_r = Int("p_r")
        self.optimizer.add(0 <= self.p_r, self.p_r < self.k_i)
        # Output trimming. See SlidingWindowParams for details.
        self.t_o = Int("t_o")
        self.optimizer.add(0 <= self.t_o, self.t_o < self.k_o)

        # Streaming params
        _, _, self.in_size_bias, self.out_size_bias, self.in_delay, self.out_delay, self.in_context_size = (
            get_streaming_params(self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o)
        )
        # FIXME!
        self.optimizer.add(
            self.in_size_bias >= 0, self.in_size_bias < self.s_i, self.in_delay >= 0, self.in_delay < self.s_i
        )

        # Blocker for guiding the solver towards simpler solutions first.
        self.solution_cost = self.k_i + self.s_i + self.p_l + self.p_r + self.k_o + self.s_o + self.t_o
        self.max_cost_sampler = ThresholdHarvester(lower_bound=4)

        # Constraints added to keep only new solutions
        self.prev_sol_constraints = []
        self.stream_param_shape_constraint = None
        self.seen_in_out_pairs = set()

        # Indicates if more solutions are available
        self.exhausted = False

    # FIXME: name
    def add_in_out_range_map(
        self,
        in_len: int,
        out_len: int,
        in_nan_range: Tuple[int, int] | None,
        out_nan_range: Tuple[int, int] | None,
    ):
        """
        TODO: doc
        """
        if in_len < 1:
            raise ValueError("The input length must be a strictly positive integer")
        if out_len < 0:
            raise ValueError("The output length must be a non-negative integer")
        if in_nan_range:
            in_nan_range = (int(in_nan_range[0]), int(in_nan_range[1]))
            if not (0 <= in_nan_range[0] < in_nan_range[1] <= in_len):
                raise ValueError("Input range must be non-empty and contained within (0, in_len)")
        if out_nan_range:
            out_nan_range = (int(out_nan_range[0]), int(out_nan_range[1]))
            if not (0 <= out_nan_range[0] < out_nan_range[1] <= out_len):
                raise ValueError("Output range must be non-empty and contained within (0, out_len), or be None")

        # Model the input to output size relation with the number of windows
        constraint_idx = len(self.optimizer.assertions())
        c = Int(f"c_{constraint_idx}")
        if (in_len, out_len) not in self.seen_in_out_pairs:
            padded_in_len = self.p_l + in_len + self.p_r
            rem = Int(f"rem_{constraint_idx}")
            self.optimizer.add(
                # Two cases: either we have enough input to get one window, either we don't
                Implies(padded_in_len < self.k_i, c == 0),
                Implies(padded_in_len >= self.k_i, c >= 1),
                # c == If(padded_in_len >= self.k_i, (padded_in_len - self.k_i) / self.s_i + 1, 0),
                Implies(
                    c >= 1,
                    And(
                        padded_in_len - self.k_i == (c - 1) * self.s_i + rem,
                        0 <= rem,
                        rem < self.s_i,
                    ),
                ),
                # Output length relation
                Implies(c == 0, out_len == 0),
                Implies(
                    c >= 1,
                    # FIXME? This doesn't seem to handle the case where we technically get a negative output size due
                    # to out trimming (out_len would still be 0 but this equality wouldn't hold?)
                    out_len == (c - 1) * self.s_o + self.k_o - 2 * self.t_o,
                ),
            )
            self.seen_in_out_pairs.add((in_len, out_len))

        # Nan trick - it has many edge cases:
        #   - Input kernels may have gaps (e.g. dilation) and thus not scan all of their inputs
        #   - Output kernels may also have gaps and thus yield disparate outputs
        #   - Nans in the input may be entirely missed because they're past the last window
        #   - Nans in the output may be entirely suppressed due to output trimming
        # As a result, the assumptions we can make are limited:
        #   - If there are multiple non-contiguous regions of nans in the input, we can't determine with certainty
        #     which region of the output results from which region of the input.
        #   - Only the first and last index of the input and output windows are guaranteed to carry over the nans, but
        #     they still may be suppressed by output trimming.
        # FIXME? Add assertions when we have no output
        if not in_nan_range or not out_nan_range:
            return

        # TODO? I could strengthen the constraints on crs/cre by putting the out nan range into a var that could also
        # be the indices trimmed by out_trim. I just need to ensure this is compatible with kernels that have gaps.

        # The window(s) that output nans must necessarily have seen a nan in their input. We'll model this.
        crs, cre = Ints(f"c_{constraint_idx}_rs c_{constraint_idx}_re")
        self.optimizer.add(
            crs <= cre,
            # crs is the index of the first window that could possibly output the first nan in the output (we have no
            # guarantee that it is indeed that window, this is a lower bound).
            crs >= 0,
            crs * self.s_o >= out_nan_range[0] - self.k_o + 1 + self.t_o,
            # Likewise, cre is the index of the last window that could possibly have output the last nan in the output.
            cre < c,
            cre * self.s_o <= out_nan_range[1] + self.t_o,
            # [crs, cre] defines a range of windows which necessarily overlaps the input nans. We have no guarantee
            # it fully contains them due to the edge cases listed above.
            self.p_l + in_nan_range[0] < cre * self.s_i + self.k_i,
            self.p_l + in_nan_range[1] >= crs * self.s_i,
        )

    def add_streamable_params(self, params: SlidingWindowParams):
        """
        If parameters were found to successfully stream the transform, this method will constrain the optimizer to
        yield only better or equally efficient solutions (in at least one aspect) with the same size parameters.
        """
        stride_in, stride_out, in_size_bias, out_size_bias, delay_in, delay_out, in_ctx = get_streaming_params(params)

        # Constraint that keeps the same in/out relation (the first 4 parameters)
        in_out_rel_constraint = And(
            self.s_i == stride_in,
            self.s_o == stride_out,
            self.in_size_bias == in_size_bias,
            self.out_size_bias == out_size_bias,
        )
        if self.stream_param_shape_constraint is None:
            self.stream_param_shape_constraint = in_out_rel_constraint

        # Constraint that ensures better or equivalent context size
        self.optimizer.add(
            Or(
                And(self.in_delay == delay_in, self.out_delay == delay_out, self.in_context_size == in_ctx),
                self.in_delay < delay_in,
                self.out_delay < delay_out,
                self.in_context_size < in_ctx,
            )
        )

    def add_non_streamable_params(self, params: SlidingWindowParams):
        """
        If parameters were found to not stream the transform, this method will refrain the optimizer from yielding
        new solutions with the same size parameters and less context.
        """
        stride_in, stride_out, in_size_bias, out_size_bias, delay_in, delay_out, in_ctx = get_streaming_params(params)

        self.optimizer.add(
            Not(
                And(
                    self.s_i == stride_in,
                    self.s_o == stride_out,
                    self.in_size_bias == in_size_bias,
                    self.out_size_bias == out_size_bias,
                    # FIXME?: It used to be possible to put <= for the delays here, but I changed the way I compute
                    # the stream offsets and now a smaller delay might actually yield a valid solution...
                    # See my personal notes for an example of this issue in practice.
                    self.in_delay == delay_in,
                    self.out_delay == delay_out,
                    self.in_context_size <= in_ctx,
                )
            )
        )

    def get_new_solution(self, valid_sols: List[SlidingWindowParams] | None = None) -> SlidingWindowParams | None:
        valid_sols = valid_sols or []

        # TODO! doc
        # TODO! iter_new_solutions, mark emitted ones, simplify constraints.

        max_cost_limit = (
            int(_MAX_COST_REL_LIMIT * max(sum(sol.as_tuple()) for sol in valid_sols))
            if valid_sols
            else _MAX_COST_FLAT_LIMIT
        )

        while True:
            max_cost_value = self.max_cost_sampler.next_p()
            max_cost_reached = max_cost_value >= max_cost_limit
            max_cost_value = min(max_cost_value, max_cost_limit)
            guide_constraints = [self.solution_cost <= max_cost_value]
            if self.stream_param_shape_constraint is not None:
                guide_constraints.append(self.stream_param_shape_constraint)

            import time

            start = time.perf_counter()
            check = self.optimizer.check(guide_constraints)
            # print(f"CHECK: {time.perf_counter() - start:.03f}s - {len(self.optimizer.assertions())} assertions")
            # print("\x1b[31m", self.optimizer.statistics(), "\x1b[39m", sep="")

            if check == sat:
                model = self.optimizer.model()
                model_values = (
                    model[self.k_i].as_long(),
                    model[self.s_i].as_long(),
                    model[self.p_l].as_long(),
                    model[self.p_r].as_long(),
                    model[self.k_o].as_long(),
                    model[self.s_o].as_long(),
                    model[self.t_o].as_long(),
                )

                # Enforce new solutions only
                new_sol_constraint = Or(
                    self.k_i != model[self.k_i],
                    self.s_i != model[self.s_i],
                    self.p_l != model[self.p_l],
                    self.p_r != model[self.p_r],
                    self.k_o != model[self.k_o],
                    self.s_o != model[self.s_o],
                    self.t_o != model[self.t_o],
                )
                self.optimizer.add(new_sol_constraint)
                self.prev_sol_constraints.append(new_sol_constraint)

                # Inform our sampler of the result
                cost = sum(model_values)
                logger.debug(f"Sampled with max cost={max_cost_value}, got solution with cost={cost}")
                self.max_cost_sampler.update(cost)

                return SlidingWindowParams(*model_values)

            else:
                logger.debug(f"Sampled with max cost={max_cost_value}, got nothing")
                self.max_cost_sampler.update(None)

                if max_cost_reached:
                    if self.stream_param_shape_constraint is not None:
                        logger.debug("Unlocking shape constraint")
                        self.stream_param_shape_constraint = None
                    else:
                        self.max_cost_sampler.update(None)
                        self.exhausted = True
                        return None

    def get_violations(self, solution: SlidingWindowParams, include_new_sol_assertions: bool = False):
        # TODO: doc
        unsat_solver = Solver()

        trackers = []
        for idx, assertion in enumerate(self.optimizer.assertions()):
            if not include_new_sol_assertions and assertion in self.prev_sol_constraints:
                continue

            bool_tracker = Bool(f"assertion_{idx}")
            unsat_solver.assert_and_track(assertion, bool_tracker)
            trackers.append((bool_tracker, assertion))

        unsat_solver.add(
            And(
                self.k_i == solution.kernel_size_in,
                self.s_i == solution.stride_in,
                self.p_l == solution.left_pad,
                self.p_r == solution.right_pad,
                self.k_o == solution.kernel_size_out,
                self.s_o == solution.stride_out,
                self.t_o == solution.out_trim,
            )
        )

        unsat_solver.check()
        violations = [
            expression for (bool_tracker, expression) in trackers if bool_tracker in unsat_solver.unsat_core()
        ]
        return violations

    def is_compatible(self, solution: SlidingWindowParams) -> bool:
        return not self.get_violations(solution)
