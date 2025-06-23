import itertools
import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from z3 import And

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.nan_trick import determine_kernel_sparsity, get_nan_map, run_nan_trick
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_sampler import SlidingWindowParamsSampler
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream, get_streaming_params
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)


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
        self.hypotheses_to_test: List[SlidingWindowParamsSolver.Hypothesis] = []
        self.rejected_hypotheses: List[SlidingWindowParamsSolver.Hypothesis] = []
        self.nan_trick_history = []
        self.validated_stream_params = set()

        # FIXME: doc & names
        self.nan_trick_params = self.get_next_nan_trick_params([])

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

    def get_next_nan_trick_params(self, hypotheses: List[Hypothesis]) -> Tuple[int, Tuple[int, int] | None] | None:
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
        hyp_stream_params = get_streaming_params(hypothesis.params)

        # FIXME!: this optim does check that the in/out size relation is identical. Is that always the case??
        #   -> If yes, can be reenabled
        # if hyp_stream_params in self.validated_stream_params:
        #     hypothesis.streaming_rejected = False
        #     return

        stride_in, stride_out, off_in, off_out, in_ctx = hyp_stream_params
        sol_stride_in, sol_stride_out, sol_off_in, sol_off_out, sol_in_ctx = self.sampler.get_streaming_params()

        # FIXME?: A justification for the number 10
        in_size = hypothesis.params.get_min_input_size_for_num_wins(10)
        in_seq = self.input_provider(in_size)

        # FIXME! not relying on a try/catch mechanism
        try:
            # TODO: clean up the streaming impl to clearly reflect that it fails if the output size is not as expected
            # TODO! use the in/out sizes generated in streaming as data
            # TODO? Cache the transform outputs
            test_stream_equivalent(
                self.trsfm,
                SlidingWindowStream(self.trsfm, hypothesis.params, in_seq.spec, self.out_spec),
                in_seq,
                atol=self.atol,
            )
            hypothesis.streaming_rejected = False
            self.validated_stream_params.add(hyp_stream_params)

            # FIXME
            logger.debug(f"Successfully streamed hypothesis {hypothesis.params}")

            # TODO: keep track of the constraint in order to be able to revert it later if the equivalence
            # test fails
            # Enforce solutions that are equally or more efficient on at least one aspect, both in the sampler
            # and in current hypotheses
            # self.sampler.optimizer.add(
            #     Implies(
            #         And(
            #             sol_stride_in == stride_in,
            #             sol_stride_out == stride_out,
            #         ),
            #         # Or(
            #         #     sol_off_in == off_in and sol_off_out == off_out and sol_in_ctx == in_ctx,
            #         #     sol_off_in < off_in,
            #         #     sol_off_out < off_out,
            #         #     sol_in_ctx < in_ctx,
            #         # ),
            #         And(sol_off_in <= off_in, sol_off_out <= off_out, sol_in_ctx <= in_ctx),
            #     )
            # )
            self.sampler.optimizer.add(
                And(
                    sol_stride_in == stride_in,
                    sol_stride_out == stride_out,
                    sol_off_in <= off_in,
                    sol_off_out <= off_out,
                    sol_in_ctx <= in_ctx,
                )
            )
            # FIXME!! discrepancy with the above: stride is not enforced
            for other_hyp in list(self.hypotheses):
                ot_stride_in, ot_stride_out, ot_off_in, ot_off_out, ot_in_ctx = get_streaming_params(other_hyp.params)
                if (
                    ot_stride_in == stride_in
                    and ot_stride_out == stride_out
                    and (ot_off_in > off_in or ot_off_out > off_out or ot_in_ctx > in_ctx)
                ):
                    other_hyp.suboptimal_rejected = True

        except AssertionError:
            hypothesis.streaming_rejected = True

            # FIXME: this cannot work if we can't tell whether the solution was rejected due to size relation or
            # context!
            # # The solution failed, let's reject solutions with the same streaming parameters and less context
            # self.sampler.optimizer.add(
            #     Not(
            #         And(
            #             sol_stride_in == stride_in,
            #             sol_stride_out == stride_out,
            #             sol_off_in <= off_in,
            #             sol_off_out <= off_out,
            #             sol_in_ctx <= in_ctx,
            #         )
            #     )
            # )

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
            self.nan_trick_params = self.get_next_nan_trick_params([])
            return

        # Update all current hypotheses, rejecting incompatible ones in the process
        for hypothesis in list(self.hypotheses):
            self.update_reject_hypotheses(hypothesis)
        if len(self.nan_trick_history) > 1:
            assert len(self.hypotheses_to_test) > len(self.hypotheses), "Internal error: no hypotheses were rejected"
        logger.info(
            f"Step {len(self.nan_trick_history)}: "
            f"rejected {len(self.hypotheses_to_test) - len(self.hypotheses)}/{len(self.hypotheses_to_test)} hypotheses"
        )

        # Get new hypotheses
        sampler_times = []
        self.hypotheses_to_test = list(self.hypotheses)
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
                print("\x1b[31m", get_streaming_params(params), "\x1b[39m", sep="")
                print("\x1b[31m", params, "\x1b[39m", sep="")
                hypothesis = SlidingWindowParamsSolver.Hypothesis(params)
                self.hypotheses_to_test.append(hypothesis)
                self.update_reject_hypotheses(hypothesis)

            # Get the next NaN trick params
            if len(self.hypotheses_to_test) >= self.max_hypotheses_per_step or self.sampler_exhausted:
                self.nan_trick_params = self.get_next_nan_trick_params(self.hypotheses_to_test)

            # FIXME accurate timing
            if params:
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
        logger.debug(self.hypotheses)

        # TODO: sort by param complexity
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
