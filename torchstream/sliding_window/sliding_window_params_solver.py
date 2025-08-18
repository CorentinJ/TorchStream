import itertools
import logging
import time
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from colorama import Fore as colors

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.kernel_sparsity import determine_kernel_sparsity, get_init_kernel_array
from torchstream.sliding_window.nan_trick import get_nan_map, get_seq_nan_idx
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
        atol: float = 1e-5,
        debug_ref_params: SlidingWindowParams | None = None,
    ):
        if isinstance(input_provider, SeqSpec):
            in_spec = input_provider
            input_provider = partial(Sequence.randn, in_spec)

        self._trsfm = trsfm
        self.input_provider = lru_cache()(input_provider)
        self.out_spec = out_spec
        self.init_seq_size = init_seq_size
        self.max_in_seq_size = max_in_seq_size
        self.atol = atol

        self.sampler = SlidingWindowParamsSampler()
        self.hypotheses: List[SlidingWindowParamsSolver.Hypothesis] = []
        self.rejected_hypotheses: List[SlidingWindowParamsSolver.Hypothesis] = []
        self.validated_streaming_params = set()
        self.nan_trick_history = []

        # FIXME!
        self.in_spec = self.input_provider(0).spec

        self.debug_ref_params = debug_ref_params

    def _trsfm_with_tracking(self, in_seq: Sequence):
        """
        Wrap the transform so that we can record in/out sizes
        """
        if not isinstance(in_seq, Sequence):
            in_seq = Sequence(self.in_spec, in_seq)

        if not in_seq.size:
            raise ValueError(f"Input sequence size must be greater than 0, got {in_seq.size}")

        # TODO!!: cleaner cache impl
        # cached_record = next(
        #     (rec for rec in self.nan_trick_history if torch.equal(rec["in_seq"].data, in_seq.data)), None
        # )
        # if cached_record:
        #     logger.debug(f"Cache hit for {in_seq.shape}, constraints={len(self.sampler.optimizer.assertions())}")
        #     v2 = cached_record["out_seq"].copy()
        #     v2.close_input()
        #     return v2

        in_nan_idx = get_seq_nan_idx(in_seq)
        # FIXME: discrepancy
        in_nan_range = (in_nan_idx[0], in_nan_idx[-1] + 1) if len(in_nan_idx) else None

        out_seq = Sequence.apply(self._trsfm, in_seq, self.out_spec, catch_zero_size_errors=True)

        out_nan_idx = get_seq_nan_idx(out_seq)
        out_nan_range = (out_nan_idx[0], out_nan_idx[-1] + 1) if len(out_nan_idx) else None

        # TODO! In/out details
        logger.debug(f"Forward {in_seq.size}->{out_seq.size} with nans {in_nan_idx}->{out_nan_idx}")

        # Keep track of the outcome in the history
        record = {
            "in_seq": in_seq.copy(),
            "in_nan_range": in_nan_range,
            "out_seq": out_seq.copy(),
            "out_nan_idx": out_nan_idx,
        }
        self.nan_trick_history.append(record)

        # Provide the results to the sampler
        self.sampler.add_in_out_range_map(in_seq.size, out_seq.size, in_nan_range, out_nan_range)
        self._debug_check_ref_params("running the transform")

        return out_seq

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

    def get_next_nan_trick_params(self) -> Tuple[int, Tuple[int, int] | None] | None:
        """
        FIXME: docstring
        Determines an input size and an input nan range for the next nan trick step.
        When hypotheses are available, this function will return parameters not used before that allows discrimating
        between at least two hypotheses. If that cannot be guaranteed, it will return None instead.
        """
        assert len(self.hypotheses) > 1

        # First, reject previously seen params, reusing them would be a waste of compute
        # FIXME: range vs idx discrepancy
        prev_seen_results = set((record["in_seq"].size, record["in_nan_range"]) for record in self.nan_trick_history)

        # If we have hypotheses, we'll determine our nan trick parameters based on them
        min_in_size = min(hyp.params.get_min_input_size() for hyp in self.hypotheses)
        # FIXME!!
        max_in_size = min_in_size + len(self.hypotheses) + 100

        best_infogain = 0.0
        # FIXME!!
        best_in_size = 100
        for in_size in range(min_in_size, max_in_size + 1):
            infogain = self._get_infogain_for_hypotheses(self.hypotheses, in_size, set())
            if infogain > best_infogain:
                best_infogain = infogain
                best_in_size = in_size

        best_nan_range = None
        for nan_idx in range(0, best_in_size):
            infogain = self._get_infogain_for_hypotheses(self.hypotheses, best_in_size, {nan_idx})
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
        if in_nan_range and not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq_size):
            raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")

        # Get an input of said size and perform the nan trick on the actual transform
        in_seq = self.input_provider(in_seq_size).copy()  # FIXME!
        if not isinstance(in_seq, Sequence):
            raise TypeError(
                f"The input_provider function {self.input_provider} returned a {type(in_seq)} "
                f"when a Sequence was expected"
            )

        # Corrupt the given range of the input sequence with NaNs
        if in_nan_range:
            in_seq[slice(*in_nan_range)] = float("nan")

        out_seq = self._trsfm_with_tracking(in_seq)

        # FIXME: duplication
        out_nan_idx = get_seq_nan_idx(out_seq)

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

        return out_seq, out_nan_idx

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
            return

        # FIXME?: A justification for the number 10
        in_size = max(50, hypothesis.params.get_min_input_size_for_num_wins(10))
        in_seq = self.input_provider(in_size).copy()  # FIXME!

        # FIXME! not relying on a try/catch mechanism
        try:
            # TODO: clean up the streaming impl to clearly reflect that it fails if the output size is not as expected
            # TODO? Cache the transform outputs
            test_stream_equivalent(
                self._trsfm_with_tracking,
                SlidingWindowStream(self._trsfm_with_tracking, hypothesis.params, in_seq.spec, self.out_spec),
                in_seq,
                atol=self.atol,
            )
            hypothesis.streaming_rejected = False
            self.validated_streaming_params.add(hyp_stream_params)

            # Enforce more efficient solutions with the same size parameters
            # FIXME: not elegant given caller
            self.sampler.add_streamable_params(hypothesis.params)
            self._debug_check_ref_params("accepting an hypothesis for streaming", hypothesis.params)
            # FIXME!! RESTORE
            # for other_hyp in list(self.hypotheses):
            #     if not self.sampler.is_compatible(other_hyp.params):
            #         other_hyp.suboptimal_rejected = True

        except IncorrectSlidingWindowParametersError:
            hypothesis.streaming_rejected = True

        except ValueError:
            hypothesis.streaming_rejected = True
            self.sampler.add_non_streamable_params(hypothesis.params)
            self._debug_check_ref_params("rejecting an hypothesis for streaming", hypothesis.params)

    # FIXME: "update" does not convey much sense
    def update_all_hypotheses(self):
        """
        FIXME: doc
        Test a hypothesis for compatibility with the transform. Updates the sampler with new constraints based on the
        outcome. If the hypothesis is accepted, it is added to the list of hypotheses. Accepting the hypothesis might
        cause other suboptimal hypotheses to be rejected in the process.
        """
        for hypothesis in self.hypotheses:
            # TODO: rename
            self.test_update_hypothesis_against_nan_trick_history(hypothesis)

            # TODO!: We test hypothesis for streaming even when they fail the kernel check because it lets us
            # validate streaming parameters. Reflect on why we do this.
            if hypothesis.streaming_rejected is None:
                self.test_update_hypothesis_by_streaming(hypothesis)

        for hypothesis in list(self.hypotheses):
            if hypothesis.rejected:
                self.hypotheses.remove(hypothesis)
                self.rejected_hypotheses.append(hypothesis)

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
                in_seq = self.input_provider(in_size).copy()  # FIXME!
                try:
                    test_stream_equivalent(
                        self._trsfm,
                        SlidingWindowStream(self._trsfm, other_params, in_seq.spec, self.out_spec),
                        in_seq,
                        atol=self.atol,
                    )
                except:
                    pass
                test_stream_equivalent(
                    self._trsfm,
                    SlidingWindowStream(self._trsfm, self.debug_ref_params, in_seq.spec, self.out_spec),
                    in_seq,
                    atol=self.atol,
                )

            self.debug_ref_params = None

    def solve(self) -> List[SlidingWindowParams]:
        # In the first part of the process, we'll forward inputs to the transform and stop as soon as we get a
        # output sequence of non-zero size
        while True:
            # Use sane defaults for the NaN trick
            out_seq, _ = self.run_nan_trick(self.init_seq_size, (self.init_seq_size // 2, self.init_seq_size // 2 + 1))
            if out_seq.size:
                break

            # As long as we haven't had a valid output, we'll increase the input size. We do this before involving
            # the sampler, otherwise we may be stuck sampling for a while before getting decent candidates.
            self.init_seq_size = min(10 * self.init_seq_size, self.max_in_seq_size)
            logger.info(
                f"Transform failed with input size {self.nan_trick_history[-1]['in_seq'].size}. "
                f"Increasing init sequence size to {self.init_seq_size}"
            )

        # In the second part, we sample hypotheses given observed inputs/outputs and continuously refine them
        sampler_times = []
        while True:
            # Sample sliding window parameters
            # FIXME more interesting timing infos
            sampler_start_time = time.perf_counter()
            params = self.sampler.get_new_solution([hyp.params for hyp in self.hypotheses])
            if params is None:
                break
            sampler_times.append(time.perf_counter() - sampler_start_time)

            # Check if the new parameters are compatible with the transform, possibly rejecting older hypotheses
            # in the process
            hypothesis = SlidingWindowParamsSolver.Hypothesis(params)
            self.hypotheses.append(hypothesis)
            self.update_all_hypotheses()

            assert self.debug_ref_params
            hyp_stream_params = get_streaming_params(hypothesis.params)
            # TODO!! self.debug_ref_params == hypothesis.params
            stream_param_comp_str = ", ".join(
                f"{p}"
                + ("" if p == p_ref else (f"{colors.RED}(>{p_ref})" if p > p_ref else f"{colors.GREEN}(<{p_ref})"))
                + colors.RESET
                for p, p_ref in zip(hyp_stream_params, get_streaming_params(self.debug_ref_params))
            )
            logger.debug(
                f"Step {len(sampler_times)}: "
                f"{'REJECTED' if hypothesis.rejected else 'ACCEPTED'} ("
                f"kernel={((colors.RED + 'FAIL') if hypothesis.nan_trick_rejected else (colors.GREEN + 'OK')) + colors.RESET}, "
                f"stream={((colors.RED + 'FAIL') if hypothesis.streaming_rejected else (colors.GREEN + 'OK')) + colors.RESET}) "
                f"new hypothesis {hypothesis.params} "
                f"with streaming params ({stream_param_comp_str})"
            )

            # In the event we now have multiple compatible hypotheses, we can search for a specific input that will
            # let us distinguish between them.
            if not hypothesis.rejected and len(self.hypotheses) > 1:
                nan_trick_params = self.get_next_nan_trick_params()
                if nan_trick_params is not None:
                    prev_n_hyps = len(self.hypotheses)
                    self.run_nan_trick(*nan_trick_params)
                    self.update_all_hypotheses()
                    assert prev_n_hyps > len(self.hypotheses), (
                        "Internal error: NaN trick did not discard any hypotheses"
                    )

                    logger.info(
                        f"Step {len(sampler_times)}: "
                        f"rejected {prev_n_hyps - len(self.hypotheses)}/{prev_n_hyps} hypotheses"
                    )

        # logger.info(
        #     f"Step {self.steps}: "
        #     # FIXME
        #     f"sampled {len(sampler_times)} new hypotheses "
        #     f"in {sum(sampler_times) * 1000:.0f}ms "
        #     f"(mean={np.mean(sampler_times) * 1000:.0f}ms), "
        # )

        logger.debug(f"Hypotheses at the end of solver execution: {self.hypotheses}")

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
    """
    return SlidingWindowParamsSolver(
        trsfm=trsfm,
        input_provider=input_provider,
        out_spec=out_spec,
        init_seq_size=init_seq_size,
        max_in_seq_size=max_in_seq_size,
        atol=atol,
        debug_ref_params=debug_ref_params,
    ).solve()
