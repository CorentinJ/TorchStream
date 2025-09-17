import itertools
import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from colorama import Fore as colors

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.kernel_sparsity import determine_kernel_sparsity, get_init_kernel_array
from torchstream.sliding_window.nan_trick import get_nan_map, get_seq_nan_idx
from torchstream.sliding_window.sliding_window_in_out_rel_sampler import (
    SlidingWindowInOutRelSampler,
    input_size_by_max_infogain,
)
from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
    get_all_output_delays,
    get_streaming_context_size,
)
from torchstream.sliding_window.sliding_window_params_sampler import (
    SlidingWindowParamsSampler,
    get_canonicalized_in_out_size_params,
)
from torchstream.sliding_window.sliding_window_stream import (
    IncorrectSlidingWindowParametersError,
    SlidingWindowStream,
)
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)


def _compare_params_str(params: tuple, real_params: tuple | None) -> str:
    assert not real_params or len(params) == len(real_params)
    if real_params is None:
        return ", ".join(map(str, params))
    if real_params != params:
        return ", ".join(
            f"{p}"
            + ("" if p == p_ref else (f"{colors.RED}(>{p_ref})" if p > p_ref else f"{colors.GREEN}(<{p_ref})"))
            + colors.RESET
            for p, p_ref in zip(params, real_params)
        )
    else:
        return colors.BLUE + ", ".join(map(str, params)) + colors.RESET


def _compare_sli_params_str(params: SlidingWindowParams, real_params: SlidingWindowParams | None = None) -> str:
    if real_params:
        ref_params = real_params.as_tuple()
        ref_shape = get_canonicalized_in_out_size_params(real_params)
        ref_delays = get_all_output_delays(real_params)
        ref_ctx = (get_streaming_context_size(real_params),)
    else:
        ref_params, ref_shape, ref_delays, ref_ctx = None, None, None, None

    return (
        f"\n\tparameters ({_compare_params_str(params.as_tuple(), ref_params)})"
        f"\n\twith shape ({_compare_params_str(get_canonicalized_in_out_size_params(params), ref_shape)})"
        f"\n\twith delays ({_compare_params_str(get_all_output_delays(params), ref_delays)})"
        f"\n\twith context size {_compare_params_str((get_streaming_context_size(params),), ref_ctx)}"
    )


class SlidingWindowParamsSolver:
    @dataclass
    class Hypothesis:
        params: SlidingWindowParams
        in_out_size_params: Tuple[int, int, int, int] | None = None
        out_delays: Tuple[int, ...] | None = None
        context_size: int | None = None
        kernels: Tuple[np.ndarray, np.ndarray] | None = None
        n_records_validated: int = 0
        # FIXME!
        nan_trick_rejected: bool = False
        delay_rejected: bool | None = None
        streaming_rejected: bool | None = None
        suboptimal_rejected: bool = False

        @property
        def rejected(self) -> bool:
            return (
                self.nan_trick_rejected
                or (self.delay_rejected is True)
                or (self.streaming_rejected is True)
                or self.suboptimal_rejected
            )

        def __post_init__(self):
            if self.kernels is None:
                self.kernels = (
                    get_init_kernel_array(self.params.kernel_size_in),
                    get_init_kernel_array(self.params.kernel_size_out),
                )

            assert self.in_out_size_params is None and self.out_delays is None and self.context_size is None
            self.in_out_size_params = get_canonicalized_in_out_size_params(self.params)
            self.out_delays = get_all_output_delays(self.params)
            self.context_size = get_streaming_context_size(self.params)

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
        max_equivalent_sols: int | None = 5,
    ):
        if isinstance(input_provider, SeqSpec):
            in_spec = input_provider
            input_provider = partial(Sequence.randn, in_spec)

        self._trsfm = trsfm
        # FIXME!
        # self.input_provider = lru_cache()(input_provider)
        self.input_provider = input_provider
        self.out_spec = out_spec
        self.init_seq_size = init_seq_size
        self.max_in_seq_size = max_in_seq_size
        self.atol = atol
        self.max_equivalent_sols = max_equivalent_sols

        self.in_out_rel_params = None
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

        return out_seq

    def _get_infogain(category_counts: Iterable[int]) -> float:
        category_counts = list(category_counts)
        infogain = math.log(sum(category_counts))
        for category_count in category_counts:
            infogain -= (category_count * math.log(category_count)) / sum(category_counts)
        return infogain

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

    def _get_equivalent_solutions_count(self, hypothesis: Hypothesis) -> int:
        assert hypothesis in self.hypotheses
        return sum(
            (
                other_hyp.in_out_size_params == hypothesis.in_out_size_params
                and other_hyp.out_delays == hypothesis.out_delays
                and other_hyp.context_size == hypothesis.context_size
            )
            for other_hyp in self.hypotheses
        )

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

    def test_hypothesis_delays(self, sampler: SlidingWindowParamsSampler, hypothesis: Hypothesis):
        # in_size = hypothesis.params.get_min_input_size_for_num_wins(10)
        # in_seq = self.input_provider(in_size).copy()  # FIXME!

        for phase, out_delay in enumerate(hypothesis.out_delays):
            # TODO: get_min_input_size_for_out_size
            min_in_size = next(
                i for i in range(1, int(1e9)) if hypothesis.params.get_metrics_for_input(i)[2] > out_delay
            )
            # Align on the phase
            pre_nan_in_size = min_in_size + (
                (hypothesis.params.stride_in - min_in_size + phase) % hypothesis.params.stride_in
            )

            _, pre_nan_n_wins, pre_nan_out_size = hypothesis.params.get_metrics_for_input(pre_nan_in_size)
            full_in_size = hypothesis.params.get_min_input_size_for_num_wins(pre_nan_n_wins + 1)

            out_seq, out_nan_idx = self.run_nan_trick(full_in_size, (pre_nan_in_size, full_in_size))
            first_nan_idx = out_nan_idx[0] if len(out_nan_idx) else None
            measured_delay = pre_nan_out_size - first_nan_idx if first_nan_idx is not None else None

            if measured_delay is None or measured_delay != out_delay:
                hypothesis.delay_rejected = True
                return

        hypothesis.delay_rejected = False

    def test_update_hypothesis_by_streaming(self, sampler: SlidingWindowParamsSampler, hypothesis: Hypothesis):
        # TODO!!
        # # If we have already validated another hypothesis with the same streaming params, we can skip any work here
        # hyp_stream_params = get_canonicalized_in_out_size_params(hypothesis.params)
        # if hyp_stream_params in self.validated_streaming_params:
        #     hypothesis.streaming_rejected = False
        #     return

        # FIXME?: A justification for the number 10
        in_size = hypothesis.params.get_min_input_size_for_num_wins(10)
        in_seq = self.input_provider(in_size).copy()  # FIXME!

        # FIXME! not relying on a try/catch mechanism
        try:
            # TODO! more elegant approach to avoiding stride in floor division issue
            step_sizes = (
                (7, 4, 12)
                + (1,) * (self.in_out_rel_params[0] + hypothesis.params.get_min_input_size_for_num_wins(1))
                + (17, 9)
            )
            test_stream_equivalent(
                # FIXME!! tracking
                self._trsfm,
                SlidingWindowStream(self._trsfm, hypothesis.params, in_seq.spec, self.out_spec),
                in_seq,
                in_step_sizes=step_sizes,
                atol=self.atol,
            )
            self.validated_streaming_params.add(hypothesis.params)
            hypothesis.streaming_rejected = False

            # Enforce more efficient solutions with the same size parameters
            sampler.add_streamable_params(hypothesis.params)
            self._debug_check_ref_params(sampler, "accepting an hypothesis for streaming", hypothesis.params)
            for other_hyp in list(self.hypotheses):
                if not sampler.is_compatible(other_hyp.params):
                    other_hyp.suboptimal_rejected = True

        except IncorrectSlidingWindowParametersError:
            # FIXME!! these arise with insufficient context size for conv mix, to fix!!
            pass

        except ValueError:
            hypothesis.streaming_rejected = True
            sampler.add_non_streamable_params(hypothesis.params)
            self._debug_check_ref_params(sampler, "rejecting an hypothesis for streaming", hypothesis.params)

    # FIXME: "update" does not convey much sense
    def update_all_hypotheses(self, sampler: SlidingWindowParamsSampler):
        """
        FIXME: doc
        Test a hypothesis for compatibility with the transform. Updates the sampler with new constraints based on the
        outcome. If the hypothesis is accepted, it is added to the list of hypotheses. Accepting the hypothesis might
        cause other suboptimal hypotheses to be rejected in the process.
        """
        for hypothesis in self.hypotheses:
            # TODO: rename
            # FIXME! poor mechanism given the tests below add to the history, leading to state issues
            self.test_update_hypothesis_against_nan_trick_history(hypothesis)

            if not hypothesis.rejected and hypothesis.delay_rejected is None:
                self.test_hypothesis_delays(sampler, hypothesis)

            # if not hypothesis.rejected and hypothesis.streaming_rejected is None:
            #     self.test_update_hypothesis_by_streaming(sampler, hypothesis)

        for hypothesis in list(self.hypotheses):
            if hypothesis.rejected:
                self.hypotheses.remove(hypothesis)
                self.rejected_hypotheses.append(hypothesis)

    def _debug_check_ref_params(
        self, sampler, event: str, other_params: SlidingWindowParams | None = None, allow_rejection: bool = False
    ):
        """
        Debugging method for checking why a good reference hypothesis gets rejected.
        """
        if (
            self.debug_ref_params
            and (violations := sampler.get_violations(self.debug_ref_params))
            and not any(hyp.params == self.debug_ref_params for hyp in self.hypotheses)
        ):
            violations_str = "\n\n\t".join(str(v) for v in violations)
            logger.debug(
                f"{colors.RED}Reference hypothesis {_compare_sli_params_str(self.debug_ref_params)} "
                f"\nbecame incompatible with "
                f"the sampler after {event}"
                f"{_compare_sli_params_str(other_params, self.debug_ref_params) if other_params else ''}\n"
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

            if allow_rejection:
                logger.debug(f"--> {colors.BLUE}Rejection is allowed{colors.RESET}")
                self.debug_ref_params = None
            else:
                raise RuntimeError()

    def run_initial_input(self) -> dict:
        # TODO! doc
        # In the first part of the process, we'll forward inputs to the transform and stop as soon as we get a
        # output sequence of non-zero size
        while not self.nan_trick_history:
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

        return self.nan_trick_history[0]

    def find_in_out_rel_params(self) -> tuple[int, int, int, int]:
        # TODO! doc
        if self.in_out_rel_params:
            return self.in_out_rel_params

        # Ensure we have at least one example input before starting
        self.run_initial_input()

        sampler = SlidingWindowInOutRelSampler()
        for record in self.nan_trick_history:
            sampler.add_in_out_size(record["in_seq"].size, record["out_seq"].size)

        real_sol = get_canonicalized_in_out_size_params(self.debug_ref_params) if self.debug_ref_params else None

        step = 1
        shape_params_hyps = []
        while not self.in_out_rel_params:
            # Sample new shape parameters
            shape_params_hyps = sampler.get_new_solutions(shape_params_hyps)
            log_str = f"[In/out rel] Step {step} params:\n\t"
            log_str += "\n\t".join(_compare_params_str(params, real_sol) for params in shape_params_hyps)
            logger.info(log_str)

            # Our sampler explores the entire space, so if we have no solution, the transform is not a sliding window.
            # If we have only one solution, it is the correct one and does not require further testing.
            if not len(shape_params_hyps):
                raise RuntimeError(
                    "Could not determine input/output size relationship for your model. This means that your model "
                    "does not behave like a sliding window. If your model is indeed a succession of sliding window "
                    "operations, you must have upsampling operations (e.g. conv transposed) followed by a downsampling "
                    "operation (e.g. conv, pool) with respective strides that cannot be expressed as a 1/x or x/1 "
                    "ratio of integers.\n"
                    "Either way, or if you believe this is a bug, opening an issue on the TorchStream repo would be "
                    "greatly appreciated: https://github.com/CorentinJ/TorchStream/issues"
                )
            if len(shape_params_hyps) == 1:
                self.in_out_rel_params = shape_params_hyps[0]
                break

            # Discriminate between hypotheses by finding an input size that will allow us to reject at least one of
            # them based on the observed output size of the relation.
            in_size, out_sizes = input_size_by_max_infogain(shape_params_hyps)
            # TODO? should we try different nan idx values here already?
            nan_idx = (in_size // 2, in_size // 2 + 1)
            out_seq, _ = self.run_nan_trick(in_size, nan_idx)
            sampler.add_in_out_size(in_size, out_seq.size)

            # Exclude solutions that do not match the observed output size
            prev_n_hyps = len(shape_params_hyps)
            shape_params_hyps = [
                params for idx, params in enumerate(shape_params_hyps) if out_sizes[idx] == out_seq.size
            ]
            assert prev_n_hyps > len(shape_params_hyps), "Internal error: did not reject any shape hypotheses"
            logger.info(
                f"[In/out rel] Step {step}: rejected {prev_n_hyps - len(shape_params_hyps)}/{prev_n_hyps} hypotheses"
            )

            step += 1

        return self.in_out_rel_params

    def find_sliding_window_params(self):
        # Start by determining the input/output size relationship, it will heavily simplify the param search to
        # know it in advance
        sampler = SlidingWindowParamsSampler(*self.find_in_out_rel_params())
        for record in self.nan_trick_history:
            out_nan_range = (
                (record["out_nan_idx"][0], record["out_nan_idx"][-1] + 1) if len(record["out_nan_idx"]) else None
            )
            sampler.add_in_out_range_map(
                record["in_seq"].size, record["out_seq"].size, record["in_nan_range"], out_nan_range
            )
            self._debug_check_ref_params(sampler, "adding previous runs")

        step = 1
        while True:
            # Sample sliding window parameters
            params = sampler.get_new_solution([hyp.params for hyp in self.hypotheses])
            if params is None:
                break

            # Check if the new parameters are compatible with the transform, possibly rejecting older hypotheses
            # in the process
            hypothesis = SlidingWindowParamsSolver.Hypothesis(params)
            self.hypotheses.append(hypothesis)
            self.update_all_hypotheses(sampler)

            logger.debug(
                f"[Sli params] Step {step}: "
                f"{'REJECTED' if hypothesis.rejected else 'ACCEPTED'} ("
                f"kernel={((colors.RED + 'FAIL') if hypothesis.nan_trick_rejected else (colors.GREEN + 'OK')) + colors.RESET}, "
                f"delay={((colors.RED + 'FAIL') if hypothesis.delay_rejected else (colors.GREEN + 'OK')) + colors.RESET}, "
                f"stream={((colors.RED + 'FAIL') if hypothesis.streaming_rejected else (colors.GREEN + 'OK')) + colors.RESET}) "
                f"{_compare_sli_params_str(hypothesis.params, self.debug_ref_params)}"
            )

            # In the event we now have multiple compatible hypotheses, we can search for a specific input that will
            # let us distinguish between them.
            if not hypothesis.rejected and len(self.hypotheses) > 1:
                nan_trick_params = self.get_next_nan_trick_params()
                if nan_trick_params is not None:
                    in_size, in_nan_range = nan_trick_params
                    prev_n_hyps = len(self.hypotheses)
                    out_seq, out_nan_idx = self.run_nan_trick(in_size, in_nan_range)

                    out_nan_range = (out_nan_idx[0], out_nan_idx[-1] + 1) if len(out_nan_idx) else None
                    sampler.add_in_out_range_map(
                        in_len=in_size,
                        out_len=out_seq.size,
                        in_nan_range=in_nan_range,
                        out_nan_range=out_nan_range,
                    )
                    self._debug_check_ref_params(sampler, "running the transform")

                    self.update_all_hypotheses(sampler)
                    assert prev_n_hyps > len(self.hypotheses), (
                        "Internal error: NaN trick did not discard any hypotheses"
                    )
                    logger.info(f"Step {step}: rejected {prev_n_hyps - len(self.hypotheses)}/{prev_n_hyps} hypotheses")

            step += 1

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
    ).find_sliding_window_params()
