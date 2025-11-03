import logging
from typing import Tuple

from opentelemetry import trace
from z3 import And, Bool, Implies, Int, Ints, Or, Solver, sat

from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
    get_all_output_delays,
    get_canonicalized_in_out_size_params,
    get_output_delay,
    get_output_delay_bounds,
    get_streaming_context_size,
    z3_ceil_div,
    z3_max,
)
from torchstream.sliding_window.threshold_harvester import ThresholdHarvester
from torchstream.sliding_window.z3_utils import z3_floor_div

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SlidingWindowParamsSampler:
    def __init__(
        self,
        stride_in: int,
        stride_out: int,
        in_size_bias_canonical: int,
        out_size_bias_canonical: int,
        minimum_input_size: int,
        solution_perf_cost_limit: int = 10_000,
    ):
        # TODO: doc

        self.optimizer = Solver()

        ## Sliding window parameters
        # Input and output strides
        self.s_i, self.s_o = stride_in, stride_out
        # Input and output kernel sizes
        # NOTE: it's technically the kernel span, i.e. the whole span of the kernel even when dilation > 1
        self.k_i, self.k_o = Ints("k_i k_o")
        # The left input padding. I have not yet seen a case where varying the left padding is useful, so we'll
        # assume it constant.
        self.p_l = Int("p_l")
        # Maximum right input padding. Unlike the left padding the actual right padding varies in practice to line
        # up with windows when stride_in > 1
        self.p_r = Int("p_r")
        # Output trimming: this many output elements are removed both on the left and right of the output. Often used
        # for transposed convolutions
        self.t_o = Int("t_o")
        self.optimizer.add(
            # It would be highly unusual to have a stride larger than the kernel size, leading to inputs being unused or
            # to gaps in the output.
            # In general, since we allow kernels with gaps, the stride is at most the largest number of consecutive
            # non-empty kernel elements
            self.k_i >= self.s_i,
            self.k_o >= self.s_o,
            # There is no point in making the padding higher than the kernel size, as it would waste compute on
            # constant values.
            0 <= self.p_l,
            self.p_l < self.k_i,
            0 <= self.p_r,
            self.p_r < self.k_i,
            # Same for output trimming, if we're discarding more than an entire kernel, then we're effectively wasting
            # inputs
            0 <= self.t_o,
            self.t_o < self.k_o,
        )

        ## Minimum input size
        self.mis = minimum_input_size
        # TODO: isolate?
        out_needed = 1 + self.t_o * 2
        num_wins_needed = z3_ceil_div(z3_max(0, out_needed - self.k_o), self.s_o) + 1
        non_padded_min_input_size = (num_wins_needed - 1) * self.s_i + self.k_i
        native_min_input_size = z3_max(1, non_padded_min_input_size - self.p_l - self.p_r)
        self.optimizer.add(native_min_input_size <= self.mis)

        ## Streaming parameters
        self.isbc, self.osbc = in_size_bias_canonical, out_size_bias_canonical
        *_, isbc, osbc = get_canonicalized_in_out_size_params(
            self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o
        )
        # FIXME: keep either?
        self.min_od, self.max_od = get_output_delay_bounds(
            self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o
        )
        self.ods = get_all_output_delays(self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o)
        # TODO? model ictx + s_i >= native_min_input_size
        self.ictx = Int("ictx")
        self.optimizer.add(
            self.ictx
            == get_streaming_context_size(self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o)
        )

        # FIXME!
        # Bounds for the input size bias: -k_i < isb <= 2 * (k_i - 1)
        # With canonicalization we have 0 <= isbc < s_i (remainder of the division of isb by s_i)
        # Bounds for the output size bias: 2 - k_o <= osb <= k_o
        # With canonicalization we have osbc = osb + (isb // s_i) * s_o
        self.optimizer.add(
            isbc == self.isbc,
            osbc == self.osbc,
            self.min_od >= 0,
            self.min_od <= self.max_od,
            self.min_od + self.s_o >= self.max_od,
        )

        # Blockers for guiding the solver towards simpler solutions first. We've got a no free lunch situation:
        #   - A transform can be expressed with multiple sliding window parameters from the same family, and while
        #   ideally we'd return the parameters that correspond to the ground truth, our procedure here is not able
        #   to discriminate for that.
        #   - The output delay and input context size are fixed for a given transform, but intermediate solutions with
        #   higher values for these will slow down the search, so guiding the search by minimizing them is good for
        #   performance.
        #   - Solutions will very large kernel sizes will be costly to verify with the kernel sparsity solver
        # Ideally we'd figure out a solver that embeds kernel sparsity directly - or if that is too costly we'd do it
        # in a separate class and focus on streaming parameters here.
        # NOTE: using optimizer.minimize() does not seem to be an option due to the logic type used
        self.simplicity_cost = self.k_i + self.k_o + (self.p_l + self.p_r) / 2 + self.t_o
        self.max_simplicity_cost_sampler = ThresholdHarvester(lower_bound=2)
        self.performance_cost = self.max_od / self.s_o + self.ictx
        self.max_performance_cost_sampler = ThresholdHarvester()
        self.solution_perf_cost_limit = solution_perf_cost_limit

        # Constraints added to keep only new solutions
        self.prev_sol_constraints = []

    # FIXME: name
    @tracer.start_as_current_span("sli_params_sampler.add_in_out_range_map")
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
        if in_len < self.mis and out_len > 0:
            raise ValueError("The input length is smaller than the minimum input size but the output length is > 0")

        # Model the input to output size relation with the number of windows
        nw = self._model_num_wins(in_len, out_len)

        # Nan trick - it has many edge cases:
        #   - Input kernels may have gaps (e.g. dilation) and thus hop over some inputs - but a proper model should
        #   have every input seen by at least one window)
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

        self._model_nan_ranges(in_len, out_len, in_nan_range, out_nan_range, nw)
        self._model_delay(in_len, out_len, in_nan_range, out_nan_range)
        self._model_context_size(in_len, out_len, in_nan_range, out_nan_range)
        self._model_kernel_out_size(out_len, out_nan_range)

    def _model_num_wins(self, in_len: int, out_len: int):
        constraint_idx = len(self.optimizer.assertions())
        nw = Int(f"nw_{constraint_idx}")
        padded_in_len = self.p_l + in_len + self.p_r
        rem = Int(f"rem_{constraint_idx}")
        self.optimizer.add(
            ## Input -> number of wins
            # Two cases: either we have enough input to get one window, either we don't
            Implies(padded_in_len < self.k_i, nw == 0),
            Implies(padded_in_len >= self.k_i, nw >= 1),
            Implies(
                nw >= 1,
                And(
                    # Division-free expression of: c = (padded_in_len - k_i) // s_i + 1,
                    padded_in_len - self.k_i == (nw - 1) * self.s_i + rem,
                    0 <= rem,
                    rem < self.s_i,
                ),
            ),
            ## Num wins -> output size
            # No windows means no output
            Implies(nw == 0, out_len == 0),
            # If we have at least one window, there are two edge cases where might still not get an output
            Implies(
                And(nw > 0, out_len == 0),
                Or(
                    # With enough output trimming
                    (nw - 1) * self.s_o + self.k_o <= 2 * self.t_o,
                    # With a minimum input size that is set larger than the native transform input size
                    # (e.g. reflect padding does this)
                    in_len < self.mis,
                ),
            ),
            # If we do have an output, we necessarily have at least one window and the following out size relation
            Implies(out_len > 0, And(nw > 0, out_len == (nw - 1) * self.s_o + self.k_o - 2 * self.t_o)),
        )

        return nw

    def _model_nan_ranges(self, in_len: int, out_len: int, in_nan_range, out_nan_range, nw):
        # TODO? I could strengthen the constraints on crs/cre by putting the out nan range into a var that could also
        # be the indices trimmed by out_trim. I just need to ensure this is compatible with kernels that have gaps.

        # The window(s) that output nans must necessarily have seen a nan in their input. We'll model this.
        constraint_idx = len(self.optimizer.assertions())
        wrs, wre = Ints(f"wrs_{constraint_idx} wre_{constraint_idx}")
        self.optimizer.add(
            wrs <= wre,
            # wrs is the index of the first window that could possibly output the first nan in the output (we have no
            # guarantee that it is indeed that window, this is a lower bound).
            wrs >= 0,
            wrs * self.s_o >= out_nan_range[0] - self.k_o + 1 + self.t_o,
            # Likewise, cre is the index of the last window that could possibly have output the last nan in the output.
            wre < nw,
            wre * self.s_o <= out_nan_range[1] + self.t_o,
            # [crs, cre] defines a range of windows which necessarily overlaps the input nans. We have no guarantee
            # it fully contains them due to the edge cases listed above.
            self.p_l + in_nan_range[0] < wre * self.s_i + self.k_i,
            self.p_l + in_nan_range[1] >= wrs * self.s_i,
        )

    def _model_delay(self, in_len: int, out_len: int, in_nan_range, out_nan_range):
        if in_nan_range[0] > 0:
            # Count how many elements lie between the first output NaNs and the expected output size of the pre-nan
            # input
            pre_nan_out_size = max(0, ((in_nan_range[0] + self.isbc) // self.s_i) * self.s_o + self.osbc)
            n_right_elems_overwritten = pre_nan_out_size - out_nan_range[0]
            # TODO: doc
            out_delay = get_output_delay(
                self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o, in_nan_range[0]
            )
            self.optimizer.add(
                # FIXME? is this at all helpful or is it redundant?
                self.min_od <= out_delay,
                out_delay <= self.max_od,
                Or(
                    # Usual case: the first nan we see in the output is the first that could be produced.
                    And(n_right_elems_overwritten >= 0, out_delay == n_right_elems_overwritten),
                    # More rare but still normal case: the output delay is technically negative
                    And(n_right_elems_overwritten < 0, self.t_o > 0, out_delay == 0),
                    # Edge case 1: the first output is a nan. Either we're in the usual case handled above, either
                    # the delay is actually larger than measured here because output nans were trimmed on the left.
                    And(
                        n_right_elems_overwritten >= 0,
                        out_nan_range[0] == 0,
                        out_delay > pre_nan_out_size,
                        out_delay <= pre_nan_out_size + self.t_o,
                        self.t_o > 0,
                    ),
                    # Edge case 2: even if the first output is not a nan, we could be missing the first nan because
                    # the output kernel could be sparse with some output trimming. It's hard to formulate strong
                    # constraints for this case, but we at least know that the gap in the output kernel needs to be
                    # larger than the first non-nan portion of the output.
                    And(
                        self.k_o >= out_nan_range[0] + 2,
                        out_delay > pre_nan_out_size,
                        out_delay <= pre_nan_out_size + self.t_o,
                        self.t_o > 0,
                    ),
                    # Edge case 3: the input kernel has gaps and skips over the input nans. In that case the delay
                    # would be underestimated using the usual case formula, so we can't say much.
                    And(
                        out_delay > n_right_elems_overwritten,
                        self.k_i >= (in_nan_range[1] - in_nan_range[0]) + 2,
                    ),
                    # NOTE: for any given set of parameters, picking a nan range larger than the input kernel size and
                    # ensuring that the pre-nan out size is larger than the output kernel size ensures that we stay
                    # in the usual case.
                ),
            )

    def _model_context_size(self, in_len: int, out_len: int, in_nan_range, out_nan_range):
        in_nan_size, out_nan_size = in_nan_range[1] - in_nan_range[0], out_nan_range[1] - out_nan_range[0]
        post_nan_in_size, post_nan_out_size = in_len - in_nan_range[1], out_len - out_nan_range[1]

        # Obtain where a stream would end if we forwarded the input sequence up to the end of the nans
        post_nan_out_size = max(0, ((in_nan_range[1] + self.isbc) // self.s_i) * self.s_o + self.osbc)
        out_delay = get_output_delay(
            self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o, in_nan_range[1]
        )
        stream_out_pos = post_nan_out_size - out_delay

        max_wins_to_drop = in_nan_range[1] // self.s_i
        wins_ctx = z3_ceil_div(z3_max(out_nan_range[1], max_wins_to_drop * self.s_o) - stream_out_pos, self.s_o)
        buffsize_lower_bound = (wins_ctx - 1) * self.s_i + 1
        buffsize_upper_bound = (wins_ctx + 1) * self.s_i - 1

        context_buffsize = in_nan_range[1] - (z3_floor_div(in_nan_range[1] - self.ictx, self.s_i) * self.s_i)

        self.optimizer.add(
            Or(
                # Usual case: the bounds are correct
                And(
                    context_buffsize >= buffsize_lower_bound,
                    context_buffsize <= buffsize_upper_bound,
                ),
                # Edge cases: we are necessarily underestimating the context
                And(
                    context_buffsize > buffsize_upper_bound,
                    Or(
                        self.k_i >= min(in_nan_range[0], in_nan_size, post_nan_in_size) + 2,
                        self.k_o >= out_nan_size + 2,
                        self.k_o >= min(out_nan_range[0], post_nan_out_size) + 2 - self.t_o,
                    ),
                ),
            )
        )

    def _model_kernel_out_size(self, out_len: int, out_nan_range):
        out_nan_size = out_nan_range[1] - out_nan_range[0]
        post_nan_out_size = out_len - out_nan_range[1]

        kernel_mod = out_nan_size % self.s_o
        self.optimizer.add(
            Or(
                self.k_o % self.s_o == kernel_mod,
                And(
                    self.k_o % self.s_o != kernel_mod,
                    self.k_o >= min(out_nan_range[0], out_nan_size, post_nan_out_size) + 2,
                    self.t_o > 0,
                ),
            )
        )

    @tracer.start_as_current_span("sli_params_sampler.get_new_solution")
    def get_new_solution(
        self,
        *cstrs,
        same_family_as: SlidingWindowParams | None = None,
        different_family_than: SlidingWindowParams | None = None,
    ) -> SlidingWindowParams | None:
        # TODO! doc

        # FIXME? Constraints are modifying the sampling direction with stateful cost limits samplers. Shouldn't be a
        # problem in practice but it's poor design.

        while True:
            max_cost_value = self.max_performance_cost_sampler.next_p()
            cost_limit_reached = max_cost_value >= self.solution_perf_cost_limit
            max_cost_value = min(max_cost_value, self.solution_perf_cost_limit)
            guide_constraints = [self.performance_cost <= max_cost_value]

            guide_constraints.extend(cstrs)

            if same_family_as:
                guide_constraints.append(
                    And(
                        *(od == delay for od, delay in zip(self.ods, same_family_as.output_delays)),
                        self.ictx == same_family_as.streaming_context_size,
                    )
                )
            if different_family_than:
                guide_constraints.append(
                    Or(
                        *(od != delay for od, delay in zip(self.ods, different_family_than.output_delays)),
                        self.ictx != different_family_than.streaming_context_size,
                    )
                )

            with tracer.start_as_current_span("sli_params_sampler.solver_check"):
                check = self.optimizer.check(guide_constraints)
            if check == sat:
                # Swap out the solution for one with minimal simplicity
                while True:
                    max_simplicity_cost = self.max_simplicity_cost_sampler.next_p()
                    if self.optimizer.check(guide_constraints + [self.simplicity_cost <= max_simplicity_cost]) == sat:
                        self.max_simplicity_cost_sampler.update(max_simplicity_cost)
                        break
                    else:
                        self.max_simplicity_cost_sampler.update(None)

                model = self.optimizer.model()
                params = SlidingWindowParams(
                    model[self.k_i].as_long(),
                    self.s_i,
                    model[self.p_l].as_long(),
                    model[self.p_r].as_long(),
                    model[self.k_o].as_long(),
                    self.s_o,
                    model[self.t_o].as_long(),
                    self.mis,
                )

                # Enforce new solutions only
                new_sol_constraint = Or(
                    self.k_i != model[self.k_i],
                    self.p_l != model[self.p_l],
                    self.p_r != model[self.p_r],
                    self.k_o != model[self.k_o],
                    self.t_o != model[self.t_o],
                )
                self.optimizer.add(new_sol_constraint)
                self.prev_sol_constraints.append(new_sol_constraint)

                # Inform our sampler of the result
                perf_cost = params.output_delay_bounds[1] // self.s_o + params.streaming_context_size
                logger.info(f"Sampled with max cost={max_cost_value:,}, got solution with cost={perf_cost:,}")
                self.max_performance_cost_sampler.update(perf_cost)

                return params

            else:
                self.max_performance_cost_sampler.update(None)

                if cost_limit_reached:
                    logger.info(f"Sampled with max cost={max_cost_value:,}, got nothing")
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

    @tracer.start_as_current_span("sli_params_sampler.is_compatible")
    def is_compatible(self, solution: SlidingWindowParams) -> bool:
        return not self.get_violations(solution)
