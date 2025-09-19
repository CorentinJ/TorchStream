import itertools
import logging
import math
from typing import Iterable, List, Tuple

from z3 import And, Bool, Implies, Int, Ints, Or, Solver, sat

from torchstream.sliding_window.nan_trick import get_nan_map
from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
    get_canonicalized_in_out_size_params,
    get_output_delay,
    get_output_delay_bounds,
    get_streaming_context_size,
)
from torchstream.sliding_window.threshold_harvester import ThresholdHarvester

logger = logging.getLogger(__name__)


class SlidingWindowParamsSampler:
    def __init__(self, stride_in: int, stride_out: int, in_size_bias_canonical: int, out_size_bias_canonical: int):
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

        ## Streaming parameters
        self.isbc, self.osbc = in_size_bias_canonical, out_size_bias_canonical
        *_, isbc, osbc = get_canonicalized_in_out_size_params(
            self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o
        )
        self.min_od, self.max_od = get_output_delay_bounds(
            self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o
        )
        self.ictx = get_streaming_context_size(self.k_i, self.s_i, self.p_l, self.p_r, self.k_o, self.s_o, self.t_o)

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
            self.ictx >= 0,
        )

        # Blocker for guiding the solver towards simpler solutions first.
        self.solution_cost = self.k_i + self.s_i + self.p_l + self.p_r + self.k_o + self.s_o + self.t_o
        self.max_cost_sampler = ThresholdHarvester(lower_bound=4)

        # Constraints added to keep only new solutions
        self.prev_sol_constraints = []
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
        # TODO: remove? The input size relation is known beforehand
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
                Implies(And(c > 0, out_len == 0), (c - 1) * self.s_o + self.k_o <= 2 * self.t_o),
                Implies(And(c > 0, out_len > 0), out_len == (c - 1) * self.s_o + self.k_o - 2 * self.t_o),
            )
            self.seen_in_out_pairs.add((in_len, out_len))

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

        if in_nan_range[0] == 0:
            return
        # Count how many elements lie between the first output NaNs and the expected output size of the pre-nan input
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
                    n_right_elems_overwritten >= 0,
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
                # NOTE: if the solution is known, picking a nan range larger than the input kernel size and
                # ensuring that the pre-nan out size is larger than the output kernel size ensures that we stay
                # in the usual case.
            ),
        )

    def add_streamable_params(self, params: SlidingWindowParams, strict=False):
        """
        If parameters were found to successfully stream the transform, this method will constrain the optimizer to
        yield only better or equally efficient solutions (in at least one aspect) with the same size parameters.
        """
        assert get_canonicalized_in_out_size_params(params) == (self.s_i, self.s_o, self.isbc, self.osbc), (
            "The input/output size parameters must match the set relation"
        )

        ictx = get_streaming_context_size(params)
        ref_min_od, ref_max_od = get_output_delay_bounds(params)

        self.optimizer.add(
            (self.ictx < ictx) if strict else (self.ictx <= ictx),
            self.min_od <= ref_min_od,
            self.max_od <= ref_max_od,
        )

    def add_non_streamable_params(self, params: SlidingWindowParams):
        """
        If parameters were found to not stream the transform, this method will refrain the optimizer from yielding
        new solutions with the same size parameters and less context.
        """
        assert get_canonicalized_in_out_size_params(params) == (self.s_i, self.s_o, self.isbc, self.osbc), (
            "The input/output size parameters must match the set relation"
        )

        ref_ictx = get_streaming_context_size(params)
        ref_min_od, ref_max_od = get_output_delay_bounds(params)

        # FIXME! review
        if ref_min_od == ref_max_od:
            self.optimizer.add(
                Or(
                    self.min_od != ref_min_od,
                    self.max_od != ref_max_od,
                    self.ictx > ref_ictx,
                )
            )

        # delay_constraint = (
        #     (self.max_od > ref_max_od)
        #     if ref_min_od == ref_max_od
        #     else And(self.min_od >= ref_min_od, self.max_od >= ref_max_od)
        # )
        # self.optimizer.add(
        #     Or(
        #         # Ensure we don't try different parameters with less context or less delay, that would not work
        #         self.ictx > ref_ictx,
        #         delay_constraint,
        #     )
        # )

    def get_new_solution(self, valid_sols: List[SlidingWindowParams] | None = None) -> SlidingWindowParams | None:
        valid_sols = valid_sols or []

        # TODO! doc
        # TODO! iter_new_solutions, mark emitted ones, simplify constraints.

        _MAX_COST_REL_LIMIT = 2.0
        _MAX_COST_FLAT_LIMIT = 10_000

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

            import time

            start = time.perf_counter()
            check = self.optimizer.check(guide_constraints)
            # print(f"CHECK: {time.perf_counter() - start:.03f}s - {len(self.sli_optimizer.assertions())} assertions")
            # print("\x1b[31m", self.sli_optimizer.statistics(), "\x1b[39m", sep="")

            if check == sat:
                model = self.optimizer.model()
                model_values = (
                    model[self.k_i].as_long(),
                    self.s_i,
                    model[self.p_l].as_long(),
                    model[self.p_r].as_long(),
                    model[self.k_o].as_long(),
                    self.s_o,
                    model[self.t_o].as_long(),
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
                cost = sum(model_values)
                logger.debug(f"Sampled with max cost={max_cost_value}, got solution with cost={cost}")
                self.max_cost_sampler.update(cost)

                return SlidingWindowParams(*model_values)

            else:
                logger.debug(f"Sampled with max cost={max_cost_value}, got nothing")
                self.max_cost_sampler.update(None)

                if max_cost_reached:
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


# FIXME!
def _get_infogain(category_counts: Iterable[int]) -> float:
    category_counts = list(category_counts)
    infogain = math.log(sum(category_counts))
    for category_count in category_counts:
        infogain -= (category_count * math.log(category_count)) / sum(category_counts)
    return infogain


def _get_infogain_for_hypotheses(all_params: List[SlidingWindowParams], input_size: int, in_nan_idx: Iterable[int]):
    # FIXME!!
    if in_nan_idx:
        in_nan_idx = next(iter(in_nan_idx))
        nan_range = (in_nan_idx, in_nan_idx + 1)
    else:
        nan_range = None

    # TODO! integrate kernels
    nan_maps = [get_nan_map(params, input_size, nan_range) for params in all_params]

    groups = [tuple(map_idx) for _, map_idx in itertools.groupby(range(len(nan_maps)), key=lambda i: len(nan_maps[i]))]
    max_out_len = max(len(nan_map) for nan_map in nan_maps)

    # FIXME!!
    if len(groups) > 1:
        return 1.0

    for out_idx in range(max_out_len):
        # new_groups = []
        # for group in groups:
        #     if len(group) == 1:
        #         new_groups.append(group)
        #         continue

        nan_values = [(nan_map[out_idx] if len(nan_map) > out_idx else None) for nan_map in nan_maps]

        unique_values = set(nan_values)
        if 1 in unique_values:
            unique_values.remove(1)
        if len(unique_values) > 1:
            return 1.0

    return 0.0


def nan_trick_params_by_max_infogain(
    all_params: List[SlidingWindowParams],
) -> Tuple[int, Tuple[int, int] | None] | None:
    """
    FIXME: docstring
    Determines an input size and an input nan range for the next nan trick step.
    When hypotheses are available, this function will return parameters not used before that allows discrimating
    between at least two hypotheses. If that cannot be guaranteed, it will return None instead.
    """
    if len(all_params) <= 1:
        return None

    # If we have hypotheses, we'll determine our nan trick parameters based on them
    min_in_size = min(params.get_min_input_size() for params in all_params)
    # FIXME!!
    max_in_size = min_in_size + len(all_params) + 100

    best_infogain = 0.0
    # FIXME!!
    best_in_size = 100
    for in_size in range(min_in_size, max_in_size + 1):
        infogain = _get_infogain_for_hypotheses(all_params, in_size, set())
        if infogain > best_infogain:
            best_infogain = infogain
            best_in_size = in_size

    best_nan_range = None
    for nan_idx in range(0, best_in_size):
        infogain = _get_infogain_for_hypotheses(all_params, best_in_size, {nan_idx})
        if infogain > best_infogain:
            best_infogain = infogain
            best_nan_range = (nan_idx, nan_idx + 1)

    # FIXME!!
    if best_infogain == 0.0:
        return None

    return best_in_size, best_nan_range
