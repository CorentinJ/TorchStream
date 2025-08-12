import itertools
import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from colorama import Fore as colors

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.kernel_sparsity import get_init_kernel_array
from torchstream.sliding_window.nan_trick import determine_kernel_sparsity, get_nan_map, run_nan_trick
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_sampler import SlidingWindowParamsSampler
from torchstream.sliding_window.sliding_window_stream import (
    IncorrectSlidingWindowParametersError,
    SlidingWindowStream,
    get_streaming_params,
)
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)


class SlidingWindowParamsSolver:
    @dataclass
    class Hypothesis:
        params: SlidingWindowParams
        kernels: Tuple[np.ndarray, np.ndarray] | None = None
        n_records_validated: int = 0
        nan_trick_rejected: bool = False
        streaming_rejected: bool | None = None
        suboptimal_rejected: bool = False

        @property
        def rejected(self) -> bool:
            return self.nan_trick_rejected or (self.streaming_rejected is True) or self.suboptimal_rejected

        def __post_init__(self):
            if self.kernels is None:
                self.kernels = (
                    get_init_kernel_array(self.params.kernel_size_in),
                    get_init_kernel_array(self.params.kernel_size_out),
                )

        def __eq__(self, other):
            if not isinstance(other, SlidingWindowParamsSolver.Hypothesis):
                return False
            return (
                self.params == other.params
                and np.array_equal(self.kernels[0], other.kernels[0])
                and np.array_equal(self.kernels[1], other.kernels[1])
            )

        def __hash__(self):
            return hash((self.params, tuple(self.kernels[0].flatten()), tuple(self.kernels[1].flatten())))

    def __init__(
        self,
        trsfm: Callable,
        input_provider: Callable[[int], Sequence] | SeqSpec,
        out_spec: SeqSpec | None = None,
        init_seq_size: int = 30,
        max_in_seq_size: int = 10_000,
        max_hypotheses_per_step: int = 20,
        atol: float = 1e-5,
        debug_ref_params: SlidingWindowParams | None = None,
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
        self.validated_streaming_params = set()
        self.nan_trick_history = []

        # FIXME: doc & names
        self.nan_trick_params = self.get_next_nan_trick_params([])

        self.debug_ref_params = debug_ref_params

        # FIXME!
        # self.hypotheses.append(SlidingWindowParamsSolver.Hypothesis(params=debug_ref_params))

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
                f"of transforms as their output kernel size is technically infinite."
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
        self._debug_check_ref_params("running the nan trick")

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
                new_kernels = determine_kernel_sparsity(
                    hypothesis.params,
                    *hypothesis.kernels,
                    record["in_seq"].size,
                    record["in_nan_range"],
                    record["out_nan_idx"],
                )
                if new_kernels[0] is None:
                    hypothesis.nan_trick_rejected = True
                    return

                # Update the hypothesis in place
                hypothesis.kernels = new_kernels

            hypothesis.n_records_validated += 1

    def test_update_hypothesis_by_streaming(self, hypothesis: Hypothesis):
        # If we have already validated another hypothesis with the same streaming params, we can skip any work here
        hyp_stream_params = get_streaming_params(hypothesis.params)
        if hyp_stream_params in self.validated_streaming_params:
            hypothesis.streaming_rejected = False
            logger.debug(f"Skipping already validated stream params for hypothesis {hypothesis.params}")
            return

        # FIXME?: A justification for the number 10
        in_size = max(50, hypothesis.params.get_min_input_size_for_num_wins(10))
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
            self.validated_streaming_params.add(hyp_stream_params)

            # FIXME
            if self.debug_ref_params is None:
                logger.debug(
                    f"Successfully streamed hypothesis {hypothesis.params} with streaming params {hyp_stream_params}"
                )
            elif self.debug_ref_params == hypothesis.params:
                logger.debug(
                    f"{colors.GREEN}Successfully streamed REFERENCE hypothesis {hypothesis.params} with "
                    f"streaming params {hyp_stream_params}{colors.RESET}"
                )
            else:
                stream_param_comp_str = ", ".join(
                    f"{p}"
                    + ("" if p == p_ref else (f"{colors.RED}(>{p_ref})" if p > p_ref else f"{colors.GREEN}(<{p_ref})"))
                    + colors.RESET
                    for p, p_ref in zip(hyp_stream_params, get_streaming_params(self.debug_ref_params))
                )
                logger.debug(
                    f"Successfully streamed DIFFERENT hypothesis {hypothesis.params} with "
                    f"streaming params ({stream_param_comp_str})"
                )

            # Enforce more efficient solutions with the same size parameters
            self.sampler.add_streamable_params(hypothesis.params)
            self._debug_check_ref_params("accepting an hypothesis for streaming", hypothesis.params)
            for other_hyp in list(self.hypotheses):
                if not self.sampler.is_compatible(other_hyp.params):
                    other_hyp.suboptimal_rejected = True

        except IncorrectSlidingWindowParametersError:
            # TODO: can we derive more constraints from this?
            hypothesis.streaming_rejected = True

        except ValueError:
            hypothesis.streaming_rejected = True
            self.sampler.add_non_streamable_params(hypothesis.params)
            self._debug_check_ref_params("rejecting an hypothesis for streaming", hypothesis.params)

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
                if other_hyp.params == self.debug_ref_params:
                    logger.debug(f"{colors.RED}Reference hypothesis {other_hyp.params} was rejected{colors.RESET}")

    def _debug_check_ref_params(self, event: str, other_params: SlidingWindowParams | None = None):
        """
        Debugging method for checking why a good reference hypothesis gets rejected.
        """
        if (
            self.debug_ref_params
            and (violations := self.sampler.get_violations(self.debug_ref_params))
            and not any(hyp.params == self.debug_ref_params for hyp in self.hypotheses)
        ):
            other_hyp_str = (
                f"Other hyp params: {other_params} with streaming params {get_streaming_params(other_params)}\n\n"
                if other_params
                else ""
            )
            violations_str = "\n\n\t".join(str(v) for v in violations)
            logger.debug(
                f"{colors.RED}Reference hypothesis {self.debug_ref_params} with streaming params "
                f"{get_streaming_params(self.debug_ref_params)}\nbecame incompatible with "
                f"the sampler after {event}:\n{other_hyp_str}"
                f"{colors.YELLOW}Violations:\n\t{violations_str}{colors.RESET}"
            )

            if other_params:
                in_size = max(50, other_params.get_min_input_size_for_num_wins(10))
                in_seq = self.input_provider(in_size)
                try:
                    test_stream_equivalent(
                        self.trsfm,
                        SlidingWindowStream(self.trsfm, other_params, in_seq.spec, self.out_spec),
                        in_seq,
                        atol=self.atol,
                    )
                except:
                    pass
                test_stream_equivalent(
                    self.trsfm,
                    SlidingWindowStream(self.trsfm, self.debug_ref_params, in_seq.spec, self.out_spec),
                    in_seq,
                    atol=self.atol,
                )

            self.debug_ref_params = None

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
        if len(self.nan_trick_history):
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
                # FIXME
                # print("\x1b[31m", get_streaming_params(params), "\x1b[39m", sep="")
                # print("\x1b[31m", params, "\x1b[39m", sep="")
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
    debug_ref_params: SlidingWindowParams | None = None,
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
        debug_ref_params=debug_ref_params,
    ).solve()
