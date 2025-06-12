import itertools
import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from z3 import And, Bool, If, Implies, Int, Ints, Or, Solver, sat

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.nan_trick import determine_kernel_sparsity, get_nan_map, run_nan_trick
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream, get_streaming_params
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)


class NoSolutionError(Exception):
    pass


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
        self.solver = Solver()
        # k_i and s_i are respectively the input kernel size and stride (NOTE: it's technically the kernel span,
        # i.e. the whole span of the kernel if dilation > 1)
        self.k_i, self.s_i = Ints("k_i s_i")
        # It would be highly unusual to have a stride larger than the kernel size, leading to inputs being dropped.
        self.solver.add(self.k_i >= self.s_i, self.s_i > 0)
        # k_o and s_o are respectively the output kernel size and stride. These are both 1 for normal convolutions,
        # but not for transposed convolutions.
        self.k_o, self.s_o = Ints("k_o s_o")
        # Again, it would be strange to have a stride larger than the kernel size, leading to gaps in the output.
        self.solver.add(self.k_o >= self.s_o, self.s_o > 0)
        # The left input padding. I have not yet seen a case where varying the left padding is useful, so we'll
        # assume it constant. Also, there is no point in making the padding higher than the kernel size, as it would
        # waste compute on constant values.
        self.p_l = Int("p_l")
        self.solver.add(0 <= self.p_l, self.p_l < self.k_i)
        # TODO doc
        self.p_r = Int("p_r")
        self.solver.add(0 <= self.p_r, self.p_r < self.k_i)
        # Output trimming. See SlidingWindowParams for details.
        self.t_o = Int("t_o")
        self.solver.add(0 <= self.t_o, self.t_o < self.k_o)

        # FIXME!!
        print("--- OUT TRIM DISABLED ---")
        self.solver.add(self.t_o == 0)

        # Blocker for guiding the solver towards simpler solutions first.
        self.cost = Int("cost")
        self.solver.add(self.k_i + self.s_i + self.p_l + self.p_r + self.k_o + self.s_o + self.t_o <= self.cost)
        self.max_cost_stack = [1_000, 100, 10]

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
        constraint_idx = len(self.solver.assertions())
        c = Int(f"c_{constraint_idx}")
        padded_in_len = self.p_l + in_len + self.p_r
        self.solver.add(
            c == If(padded_in_len >= self.k_i, (padded_in_len - self.k_i) / self.s_i + 1, 0),
            out_len == If(c > 0, (c - 1) * self.s_o + self.k_o - 2 * self.t_o, 0),
        )

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
        self.solver.add(
            crs <= cre,
            # crs is the index of the first window that could possibly output the first nan in the output (we have no
            # guarantee that it is indeed that window, this is a lower bound).
            crs >= 0,
            crs >= (out_nan_range[0] - self.k_o + 1 + self.t_o) / self.s_o,
            # Likewise, cre is the index of the last window that could possibly have output the last nan in the output.
            cre < c,
            cre <= (out_nan_range[1] + self.t_o) / self.s_o,
            # [crs, cre] defines a range of windows which necessarily overlaps the input nans. We have no guarantee
            # it fully contains them due to the edge cases listed above.
            self.p_l + in_nan_range[0] < cre * self.s_i + self.k_i,
            self.p_l + in_nan_range[1] >= crs * self.s_i,
        )

        # if not in_nan_range:
        #     return
        # padded_nan_start, padded_nan_stop = self.p_l + in_nan_range[0], self.p_l + in_nan_range[1]
        # if out_nan_range:
        #     # The start of both the input and the output range correspond to the same window. The same can be said
        #     # for the end of the ranges.
        #     # FIXME: notation difference: c above is the number of windows, cs and ce are window indices
        #     crs, cre = Ints(f"c_{constraint_idx}_rs c_{constraint_idx}_re")
        #     self.solver.add(0 <= crs, crs <= cre, cre < c)

        #     self.solver.add(out_nan_range[0] == crs * self.s_o - self.t_o)
        #     self.solver.add(
        #         crs == (If(padded_nan_start >= self.k_i, padded_nan_start - self.k_i + 1, 0) + self.s_i - 1) / self.s_i
        #     )

        #     self.solver.add(out_nan_range[1] == cre * self.s_o + self.k_o - self.t_o)
        #     self.solver.add(cre == If(padded_nan_stop > (c - 1) * self.s_i, c - 1, (padded_nan_stop - 1) / self.s_i))
        # else:
        #     # When there's an output but no in->out range, it necessarily means that the input range was dropped
        #     # due to windows not lining up. Therefore, the input range is fully contained after the last window.
        #     # FIXME!! not the case with output trimming
        #     self.solver.add(Implies(c > 0, padded_nan_start >= self.k_i + self.s_i * (c - 1)))

    def get_streaming_params(self):
        """
        This is a 1-to-1 equivalent to torchstream.sliding_window.sliding_window_stream.get_streaming_params(), for
        expressing new constraints.
        """
        n_left_wins_wasted = (self.p_l + self.s_i - 1) / self.s_i
        n_overlapping_out_wins = (self.k_o + self.s_o - 1) / self.s_o - 1
        n_wins_left_context = n_left_wins_wasted + n_overlapping_out_wins
        n_elems_right_context = self.p_r
        eff_size_bias = self.p_l - self.k_i
        elem_in_to_win_ratio = self.s_i
        win_to_elem_out_ratio = self.s_o
        return n_wins_left_context, n_elems_right_context, eff_size_bias, elem_in_to_win_ratio, win_to_elem_out_ratio

    def get_new_solution(self) -> SlidingWindowParams | None:
        # TODO! doc
        # TODO! iter_new_solutions, mark emitted ones, simplify constraints.

        while True:
            constraint = [self.cost <= self.max_cost_stack[-1]] if self.max_cost_stack else []
            if self.solver.check(constraint) == sat:
                model = self.solver.model()
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
                self.solver.add(
                    Or(
                        self.k_i != model[self.k_i],
                        self.s_i != model[self.s_i],
                        self.p_l != model[self.p_l],
                        self.p_r != model[self.p_r],
                        self.k_o != model[self.k_o],
                        self.s_o != model[self.s_o],
                        self.t_o != model[self.t_o],
                    )
                )

                # Temporarily enforce simpler solutions
                cost = sum(model_values)
                if not self.max_cost_stack or self.max_cost_stack[-1] > cost:
                    self.max_cost_stack.append(cost)

                return SlidingWindowParams(*model_values)
            elif constraint:
                self.max_cost_stack.pop(-1)
            else:
                return None

    def get_violations(self, solution: SlidingWindowParams):
        # TODO: doc
        unsat_solver = Solver()

        trackers = []
        for idx, assertion in enumerate(self.solver.assertions()):
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


class SlidingWindowParamsSolver:
    @dataclass
    class Hypothesis:
        params: SlidingWindowParams
        n_records_validated: int = 0
        nan_trick_rejected: bool = False
        streaming_rejected: bool | None = None
        suboptimal_rejected: bool = False

        @property
        def rejected(self) -> bool:
            return self.nan_trick_rejected or (self.streaming_rejected is True) or self.suboptimal_rejected

    def __init__(
        self,
        trsfm: Callable,
        input_provider: Callable[[int], Sequence] | SeqSpec,
        out_spec: SeqSpec | None = None,
        init_seq_size: int = 30,
        max_in_seq_size: int = 10_000,
        max_hypotheses_per_step: int = 20,
        atol: float = 1e-5,
    ):
        if max_hypotheses_per_step <= 1:
            raise ValueError("max_hypotheses_per_step must be greater than 1")
        if isinstance(input_provider, SeqSpec):
            in_spec = input_provider
            input_provider = partial(Sequence.randn, in_spec)

        self.trsfm = trsfm
        self.input_provider = input_provider
        self.out_spec = out_spec
        self.init_seq_size = init_seq_size
        self.max_in_seq_size = max_in_seq_size
        self.max_hypotheses_per_step = max_hypotheses_per_step
        self.atol = atol

        self.sampler = SlidingWindowParamsSampler()
        self.sampler_exhausted = False
        self.hypotheses: List[SlidingWindowParamsSolver.Hypothesis] = []
        self.rejected_hypotheses: List[SlidingWindowParamsSolver.Hypothesis] = []
        self.nan_trick_history = []

        # FIXME: doc & names
        self.nan_trick_params = self.get_best_nan_trick_params_for_hypotheses([])
        self.prev_n_rejected = 0

    # def _get_infogain(category_counts: Iterable[int]) -> float:
    #     category_counts = list(category_counts)
    #     infogain = math.log(sum(category_counts))
    #     for category_count in category_counts:
    #         infogain -= (category_count * math.log(category_count)) / sum(category_counts)
    #     return infogain

    @staticmethod
    def _get_infogain_for_hypotheses(hypotheses: List[Hypothesis], input_size: int, in_nan_idx: Iterable[int]):
        # FIXME!!
        if in_nan_idx:
            in_nan_idx = next(iter(in_nan_idx))
            nan_range = (in_nan_idx, in_nan_idx + 1)
        else:
            nan_range = None
        nan_maps = [get_nan_map(hyp.params, input_size, nan_range) for hyp in hypotheses]

        groups = [
            tuple(map_idx) for _, map_idx in itertools.groupby(range(len(nan_maps)), key=lambda i: len(nan_maps[i]))
        ]
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

    def get_best_nan_trick_params_for_hypotheses(
        self, hypotheses: List[Hypothesis]
    ) -> Tuple[int, Tuple[int, int] | None] | None:
        """
        Determines an input size and an input nan range for the next nan trick step.
        When hypotheses are available, this function will return parameters not used before that allows discrimating
        between at least two hypotheses. If that cannot be guaranteed, it will return None instead.
        """
        # First, reject previously seen params, reusing them would be a waste of compute
        # FIXME: range vs idx discrepancy
        prev_seen_results = set((record["in_seq"].size, record["in_nan_range"]) for record in self.nan_trick_history)

        # In the absence of input/output information, use sane defaults
        if not hypotheses:
            nan_trick_params = (self.init_seq_size, (self.init_seq_size // 2, self.init_seq_size // 2 + 1))
            assert nan_trick_params not in prev_seen_results, "Internal error"
            return nan_trick_params

        # If we have hypotheses, we'll determine our nan trick parameters based on them
        min_in_size = min(hyp.params.get_min_input_size() for hyp in hypotheses)
        # FIXME!!
        max_in_size = min_in_size + len(hypotheses) + 100

        best_infogain = 0.0
        # FIXME!!
        best_in_size = 100
        for in_size in range(min_in_size, max_in_size + 1):
            infogain = self._get_infogain_for_hypotheses(hypotheses, in_size, set())
            if infogain > best_infogain:
                best_infogain = infogain
                best_in_size = in_size

        best_nan_range = None
        for nan_idx in range(0, best_in_size):
            infogain = self._get_infogain_for_hypotheses(hypotheses, best_in_size, {nan_idx})
            if infogain > best_infogain:
                in_nan_range = (nan_idx, nan_idx + 1)
                if (best_in_size, in_nan_range) not in prev_seen_results:
                    best_infogain = infogain
                    best_nan_range = in_nan_range

        # FIXME!!
        if best_infogain == 0.0:
            return None

        return best_in_size, best_nan_range

    def run_nan_trick(self, in_seq_size: int, in_nan_range: Tuple[int, int] | None):
        """
        Runs the nan trick once on the transform, updating the sampler and history in the process.
        """
        # Get an input of said size and perform the nan trick on the actual transform
        in_seq = self.input_provider(in_seq_size)
        if not isinstance(in_seq, Sequence):
            raise TypeError(
                f"The input_provider function {self.input_provider} returned a {type(in_seq)} "
                f"when a Sequence was expected"
            )
        out_seq, out_nan_idx = run_nan_trick(self.trsfm, in_seq, in_nan_range, out_spec=(self.out_spec or in_seq.spec))

        # Raise if we get no output with the maximum input size
        if in_seq_size == self.max_in_seq_size and out_seq.size == 0:
            # TODO: better message
            raise RuntimeError()

        # Raise if we get all NaNs in the output with the maximum input size (kernels with infinite output size)
        # NOTE: this does not take into account the input nan range
        if in_seq_size == self.max_in_seq_size and len(out_nan_idx) == out_seq.size:
            # TODO: offer a course of action
            raise RuntimeError(
                f"Your transform outputs NaNs covering the entire output (size={out_seq.size}) given the "
                f"maximum input size (={self.max_in_seq_size}) and NaNs at {in_nan_range}. This likely means that "
                f"an operation in your transform broadcasts an input element to all output elements, like a mean, "
                f"batchnorm, etc... We can't determine sliding window parameters nor stream exactly these types "
                f"of transforms as their kernel size is technically infinite."
            )

        # Keep track of the outcome in the history
        record = {
            "step": len(self.nan_trick_history) + 1,
            "in_seq": in_seq.copy(),
            "in_nan_range": in_nan_range,
            "out_seq": out_seq.copy(),
            "out_nan_idx": out_nan_idx,
        }
        self.nan_trick_history.append(record)

        # Provide the nan trick results to the sampler
        out_nan_range = (out_nan_idx[0], out_nan_idx[-1] + 1) if len(out_nan_idx) else None
        self.sampler.add_in_out_range_map(in_seq_size, out_seq.size, in_nan_range, out_nan_range)

        return in_seq_size, in_nan_range, out_seq.size, out_nan_idx

    def test_update_hypothesis_against_nan_trick_history(self, hypothesis: Hypothesis):
        for record in self.nan_trick_history[hypothesis.n_records_validated :]:
            # Reject the hypothesis if we get a different output length
            _, _, expected_out_len = hypothesis.params.get_metrics_for_input(record["in_seq"].size)
            if record["out_seq"].size != expected_out_len:
                hypothesis.nan_trick_rejected = True
                return

            # Reject if the nan trick's output is not compatible with the hypothesis
            if record["in_nan_range"] is not None:
                kernel_in, kernel_out = determine_kernel_sparsity(
                    hypothesis.params,
                    record["in_seq"].size,
                    record["in_nan_range"],
                    record["out_nan_idx"],
                )
                if kernel_in is None or kernel_out is None:
                    hypothesis.nan_trick_rejected = True
                    return

                # Update the hypothesis in place
                hypothesis.params.kernel_in_sparsity = kernel_in
                hypothesis.params.kernel_out_sparsity = kernel_out

            hypothesis.n_records_validated += 1

    def test_update_hypothesis_by_streaming(self, hypothesis: Hypothesis):
        # FIXME!: get input size to have 10 windows instead, also clean up the streaming impl to clearly reflect that
        # it fails if the output size is not as expected
        in_seq = self.input_provider(100)

        nwlc, nerc, esb, eitwr, wteor = get_streaming_params(hypothesis.params)
        sol_nwlc, sol_nerc, sol_esb, sol_eitwr, sol_wteor = self.sampler.get_streaming_params()

        # FIXME! not relying on a try/catch mechanism
        try:
            # TODO! use the in/out sizes generated in streaming as data
            test_stream_equivalent(
                self.trsfm,
                SlidingWindowStream(self.trsfm, hypothesis.params, in_seq.spec, self.out_spec),
                in_seq,
                atol=self.atol,
            )
            hypothesis.streaming_rejected = False

            # FIXME
            logger.debug(f"Successfully streamed hypothesis {hypothesis.params}")

            # TODO: keep track of the constraint in order to be able to revert it later if the equivalence
            # test fails
            # Enforce solutions that are equally or more efficient on at least one aspect, both in the sampler
            # and in current hypotheses
            self.sampler.solver.add(
                Implies(
                    And(sol_eitwr == eitwr, sol_wteor == wteor),
                    Or(
                        And(sol_nwlc == nwlc, sol_nerc == nerc, sol_esb == esb),
                        sol_nwlc < nwlc,
                        sol_nerc < nerc,
                        sol_esb > esb,
                    ),
                )
            )
            for other_hyp in list(self.hypotheses):
                ot_nwlc, ot_nerc, ot_esb, ot_eitwr, ot_wteor = get_streaming_params(other_hyp.params)
                if (ot_eitwr == eitwr and ot_wteor == wteor) and not (
                    (ot_nwlc == nwlc and ot_nerc == nerc and ot_esb == esb)
                    or ot_nwlc < nwlc
                    or ot_nerc < nerc
                    or ot_esb > esb
                ):
                    other_hyp.suboptimal_rejected = True

        except AssertionError:
            hypothesis.streaming_rejected = True

            # The solution failed, let's reject solutions with the same streaming parameters
            self.sampler.solver.add(
                Or(
                    sol_nwlc != nwlc,
                    sol_nerc != nerc,
                    sol_esb != esb,
                    sol_eitwr != eitwr,
                    sol_wteor != wteor,
                )
            )

    def update_reject_hypotheses(self, hypothesis: Hypothesis):
        """
        Test a hypothesis for compatibility with the transform. Updates the sampler with new constraints based on the
        outcome. If the hypothesis is accepted, it is added to the list of hypotheses. Accepting the hypothesis might
        cause other suboptimal hypotheses to be rejected in the process.
        """
        if not hypothesis.rejected:
            self.test_update_hypothesis_against_nan_trick_history(hypothesis)

        if hypothesis.streaming_rejected is None:
            self.test_update_hypothesis_by_streaming(hypothesis)

        if hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)
        for other_hyp in list(self.hypotheses):
            if other_hyp.rejected:
                self.hypotheses.remove(other_hyp)
                self.rejected_hypotheses.append(other_hyp)

    def step(self):
        # Run the nan trick
        assert self.nan_trick_params is not None, "Internal error: no nan trick params available"
        self.run_nan_trick(*self.nan_trick_params)

        # If we have had no output yet, we'll quickly increase the input size before involving the sampler, or
        # we may be stuck sampling for a while before getting decent candidates.
        if all(record["in_seq"].size == 0 for record in self.nan_trick_history):
            self.init_seq_size = min(10 * self.init_seq_size, self.max_in_seq_size)
            logger.info(
                f"Transform failed with input size {self.nan_trick_history[-1]['in_seq'].size}. "
                f"Increasing init sequence size to {self.init_seq_size}"
            )
            self.nan_trick_params = self.get_best_nan_trick_params_for_hypotheses([])
            return

        # Update all current hypotheses, rejecting incompatible ones in the process
        for hypothesis in list(self.hypotheses):
            self.update_reject_hypotheses(hypothesis)
        if len(self.nan_trick_history) > 1:
            assert len(self.rejected_hypotheses) > self.prev_n_rejected, "Internal error: no hypotheses were rejected"
        self.prev_n_rejected = len(self.rejected_hypotheses)

        # Get new hypotheses
        sampler_times = []
        infogain_hypotheses = list(self.hypotheses)
        self.nan_trick_params = None
        while self.nan_trick_params is None and not self.sampler_exhausted:
            # Sample sliding window parameters
            sampler_start_time = time.perf_counter()
            params = self.sampler.get_new_solution()
            if params is None:
                self.sampler_exhausted = True

            # Validate them. Regardless of the outcome, we will add them to the hypotheses for infogain in order
            # to steer the sampler towards more promising candidates.
            if params:
                hypothesis = SlidingWindowParamsSolver.Hypothesis(params)
                infogain_hypotheses.append(hypothesis)
                self.update_reject_hypotheses(hypothesis)

            # Get the next NaN trick params
            if len(infogain_hypotheses) >= self.max_hypotheses_per_step or self.sampler_exhausted:
                self.nan_trick_params = self.get_best_nan_trick_params_for_hypotheses(infogain_hypotheses)

            # FIXME accurate timing
            sampler_times.append(time.perf_counter() - sampler_start_time)

        if sampler_times:
            logger.info(
                f"Step {len(self.nan_trick_history)}: "
                # FIXME
                f"sampled {len(sampler_times)} new hypotheses "
                f"in {sum(sampler_times) * 1000:.0f}ms "
                f"(mean={np.mean(sampler_times) * 1000:.0f}ms), "
            )

    def solve(self) -> List[SlidingWindowParams]:
        while self.nan_trick_params is not None:
            self.step()
        return [hypothesis.params for hypothesis in self.hypotheses]


# TODO: allow transforms with multiple sequential inputs
#   -> Or simply call the function multiple times? unsure
@torch.no_grad()
def find_sliding_window_params_for_transform(
    trsfm: Callable,
    input_provider: Callable[[int], Sequence] | SeqSpec,
    out_spec: SeqSpec | None = None,
    init_seq_size: int = 30,
    max_in_seq_size: int = 10_000,
    max_hypotheses_per_step: int = 20,
    atol: float = 1e-5,
) -> List[SlidingWindowParams]:
    """
    Given a sequence-to-sequence transform (neural net, single layer, time series analysis function, ...),
    this function will empirically determine sliding window parameters that correspond to the transform. That is, if
    the transform can be decomposed into a sliding window and a kernel applied to each window. This allows for
    deriving metrics for the transform (time to first output, latency, context size, chunk size needed for
    streaming, ...) as well as making the transform trivially streamable without approximations.

    This is only possible if the transform can be assimilated to a sliding window operation TODO: describe properties
    of this operation

    TODO: rewrite docs

    TODO: handle multi-input/output
    :param input_spec: specification for the input format of the transform. The transform must accept the data format
    described in the input spec as positional arguments.
    :param output_spec: same as input_spec but for the output of the transform. If the transform has multiple
    sequential outputs, they must be returned as an iterable matching the output spec. If the output spec is
    identical to the input spec, it can be omitted, and the input spec will be used instead.
    :param input_provider: a function that takes an integer representing the sequence size, and returns a Sequence of
    this size.
    FIXME: incorrect doc
    :param max_hypotheses_per_step: the sampler finds up to this amount of sliding windows parameters compatible with
    observations made over the execution of this function. Increasing this value will decrease the number of executions
    of the transforms, but increase the execution time of the sampler. If necessary, tune it according to your model's
    performances.
    """
    return SlidingWindowParamsSolver(
        trsfm=trsfm,
        input_provider=input_provider,
        out_spec=out_spec,
        init_seq_size=init_seq_size,
        max_in_seq_size=max_in_seq_size,
        max_hypotheses_per_step=max_hypotheses_per_step,
        atol=atol,
    ).solve()
