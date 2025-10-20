import logging
from functools import partial
from itertools import zip_longest
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from colorama import Fore as colors

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.kernel_sparsity import determine_kernel_sparsity, get_init_kernel_array
from torchstream.sliding_window.nan_trick import get_nan_idx
from torchstream.sliding_window.sliding_window_in_out_rel_sampler import (
    SlidingWindowInOutRelSampler,
    compute_in_to_out_sizes,
    input_size_by_max_infogain,
)
from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
    get_canonicalized_min_in_size,
    get_output_delay_bounds,
)
from torchstream.sliding_window.sliding_window_params_sampler import SlidingWindowParamsSampler
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)


def _compare_params_str(params: tuple, real_params: tuple | None, names: Iterable[str] | None = None) -> str:
    assert not real_params or len(params) == len(real_params)
    names = [n + "=" for n in names] if names else [""] * len(params)
    assert len(params) == len(names), (params, names)

    if real_params is None:
        return ", ".join(f"{name}{p}" for p, name in zip(params, names))
    if real_params != params:
        return ", ".join(
            f"{name}{p}"
            + ("" if p == p_ref else (f"{colors.RED}(>{p_ref})" if p > p_ref else f"{colors.GREEN}(<{p_ref})"))
            + colors.RESET
            for p, p_ref, name in zip(params, real_params, names)
        )
    else:
        return colors.BLUE + ", ".join(f"{name}{p}" for p, name in zip(params, names)) + colors.RESET


# TODO: I need this fancy function elsewhere, it's very useful
def _compare_sli_params_str(params: SlidingWindowParams, real_params: SlidingWindowParams | None = None) -> str:
    if real_params:
        ref_params = real_params.as_tuple(with_min_in_size=False)
        ref_shape = real_params.canonicalized_in_out_size_params
        ref_delays = real_params.output_delays
        ref_ctx = (real_params.streaming_context_size,)
    else:
        ref_params, ref_shape, ref_delays, ref_ctx = None, None, None, None

    return (
        f"\n\tparameters ({_compare_params_str(params.as_tuple(with_min_in_size=False), ref_params, 'ki,si,lp,rp,ko,so,ot'.split(','))})"
        f"\n\twith shape ({_compare_params_str(params.canonicalized_in_out_size_params, ref_shape, 's_i,s_o,isbc,osbc,mis'.split(','))})"
        f"\n\twith delays ({_compare_params_str(params.output_delays, ref_delays)})"
        f"\n\twith context size {_compare_params_str((params.streaming_context_size,), ref_ctx)}"
    )


class _SliHypothesis:
    def __init__(self, params: SlidingWindowParams, id: int):
        self.params = params
        self.id = id
        self.kernels = (
            get_init_kernel_array(self.params.kernel_size_in),
            get_init_kernel_array(self.params.kernel_size_out),
        )


class SlidingWindowParamsSolver:
    def __init__(
        self,
        trsfm: Callable,
        input_provider: Callable[[int], Sequence] | SeqSpec,
        out_spec: SeqSpec | None = None,
        init_seq_size: int = 30,
        max_in_seq_size: int = 10_000,
        atol: float = 1e-5,
        max_equivalent_sols: int = 1,
        zero_size_exception_types: Tuple[type[Exception], ...] = (RuntimeError,),
        debug_ref_params: SlidingWindowParams | None = None,
    ):
        if isinstance(input_provider, SeqSpec):
            in_spec = input_provider
            input_provider = partial(Sequence.randn, in_spec)

        self._trsfm = trsfm
        self.input_provider = input_provider
        self.out_spec = out_spec
        self.init_seq_size = init_seq_size
        self.max_in_seq_size = max_in_seq_size
        self.atol = atol
        self.max_equivalent_sols = max_equivalent_sols
        self.zero_size_exception_types = zero_size_exception_types

        self.in_out_rel_params = None
        self.nan_trick_history = []

        # FIXME!
        self.in_spec = self.input_provider(0).spec

        self.debug_ref_params = debug_ref_params
        if debug_ref_params:
            logger.info(f"Debug reference parameters: {_compare_sli_params_str(debug_ref_params)}")

    def run_nan_trick(self, in_seq_size: int, in_nan_range: Tuple[int, int] | None) -> dict:
        """
        Runs the nan trick once on the transform, updating the sampler and history in the process.
        """
        if in_nan_range and not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq_size):
            raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")

        # Running the same nan trick twice is a waste of compute. Callers are expected not to do this.
        assert not any(
            (in_seq_size, in_nan_range) == (record["in_seq_size"], record["in_nan_range"])
            for record in self.nan_trick_history
        ), "Internal error: reusing previously seen NaN trick parameters"

        # Get an input of said size and perform the nan trick on the actual transform
        in_seq = self.input_provider(in_seq_size)
        if not isinstance(in_seq, Sequence):
            raise TypeError(
                f"The input_provider function {self.input_provider} returned a {type(in_seq)} "
                f"when a Sequence was expected"
            )

        # Corrupt the given range of the input sequence with NaNs
        if in_nan_range:
            in_seq[slice(*in_nan_range)] = float("nan")

        out_seq = Sequence.apply(
            self._trsfm, in_seq, self.out_spec, zero_size_exception_types=self.zero_size_exception_types
        )

        # Keep track of the outcome in the history
        out_nan_idx = get_nan_idx(out_seq)
        out_nan_range = (int(out_nan_idx[0]), int(out_nan_idx[-1] + 1)) if len(out_nan_idx) else None
        logger.info(f"Forwarded {in_seq.size}->{out_seq.size} with nans {in_nan_range}->{out_nan_range}")
        record = {
            "in_seq_size": in_seq.size,
            "in_nan_range": in_nan_range,
            "out_seq_size": out_seq.size,
            "out_nan_idx": out_nan_idx,
            "out_nan_range": out_nan_range,
        }
        self.nan_trick_history.append(record)

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

        return record

    def run_initial_input(self) -> dict:
        # TODO! doc
        # In the first part of the process, we'll forward inputs to the transform and stop as soon as we get a
        # output sequence of non-zero size
        while True:
            # Use sane defaults for the NaN trick
            record = self.run_nan_trick(self.init_seq_size, (self.init_seq_size // 2, self.init_seq_size // 2 + 1))
            if record["out_seq_size"]:
                break

            # As long as we haven't had a valid output, we'll increase the input size. We do this before involving
            # the sampler, otherwise we may be stuck sampling for a while before getting decent candidates.
            self.init_seq_size = min(10 * self.init_seq_size, self.max_in_seq_size)
            logger.info(
                f"Transform failed with input size {record['in_seq_size']}. "
                f"Increasing init sequence size to {self.init_seq_size}"
            )

        return self.nan_trick_history[0]

    def find_in_out_rel_params(self) -> Tuple[int, int, int, int, int]:
        # TODO! doc
        if self.in_out_rel_params:
            return self.in_out_rel_params

        # Ensure we have at least one example input before starting
        self.run_initial_input()

        sampler = SlidingWindowInOutRelSampler()
        for record in self.nan_trick_history:
            sampler.add_in_out_size(record["in_seq_size"], record["out_seq_size"])

        real_sol = self.debug_ref_params.canonicalized_in_out_size_params if self.debug_ref_params else None

        step = 1
        shape_params_hyps = []
        while not self.in_out_rel_params:
            # Sample new shape parameters
            # TODO: bench values other than 2 for max sols
            shape_params_hyps = sampler.get_new_solutions(shape_params_hyps, max_sols=5)
            log_str = f"[In/out rel] Step {step} params:\n  "
            log_str += "\n  ".join(
                _compare_params_str(params, real_sol, "s_i,s_o,isbc,osbc,mis".split(","))
                for params in shape_params_hyps
            )
            logger.info(log_str)

            # Our sampler explores the entire space, so if we have no solution, the transform is not a sliding window.
            # If we have only one solution, it is the correct one and does not require further testing.
            if not len(shape_params_hyps):
                raise RuntimeError(
                    "Could not determine input/output size relationship for your transform. This means that your "
                    "transform does not behave like a sliding window. "
                    # TODO: this is rarely going to be the case for the users that get this message... Adapt
                    "\nIf your transform is a model that is indeed a succession of sliding window "
                    "operations, you must avoid upsampling operations (e.g. conv transposed) followed by a "
                    "downsampling operation (e.g. conv, pool) with respective strides that cannot be expressed as "
                    "a 1/x or x/1 ratio of integers."
                    "\nEither way, or if you believe this is a bug, opening an issue on the TorchStream repo would be "
                    "greatly appreciated: https://github.com/CorentinJ/TorchStream/issues"
                )
            if len(shape_params_hyps) == 1:
                self.in_out_rel_params = shape_params_hyps[0]
                break

            # Obtain the input size -> output size map for all hypotheses
            in_to_out_sizes = compute_in_to_out_sizes(shape_params_hyps, max_input_size=self.max_in_seq_size)

            # Pick an input size to test
            if all(params[:4] == shape_params_hyps[0][:4] for params in shape_params_hyps):
                # Heuristic: if all hypotheses have the same (s_i, s_o, isbc, osbc)
                #   - If the canonical min input size hasn't been tested, we'll test it
                #   - Otherwise we'll bisect
                canon_min_in_size = get_canonicalized_min_in_size(*shape_params_hyps[0][:4])
                if not any(record["in_seq_size"] == canon_min_in_size for record in self.nan_trick_history):
                    in_size = canon_min_in_size
                else:
                    lower_bound = max(
                        (record["in_seq_size"] for record in self.nan_trick_history if record["out_seq_size"] == 0),
                        default=canon_min_in_size,
                    )
                    upper_bound = min(
                        (record["in_seq_size"] for record in self.nan_trick_history if record["out_seq_size"] > 0),
                        default=self.max_in_seq_size + 1,
                    )
                    in_size = (lower_bound + upper_bound) // 2
            else:
                # Discriminate between hypotheses by finding an input size that will allow us to reject at least one of
                # them based on the observed output size of the relation.
                in_size = input_size_by_max_infogain(in_to_out_sizes)

            # TODO? should we try different nan idx values here already?
            #   -> Yes! That would help with converging towards solutions faster. Determine a heuristic size based
            #      on the input size relations
            nan_idx = (in_size // 2, in_size // 2 + 1)
            record = self.run_nan_trick(in_size, nan_idx)
            sampler.add_in_out_size(in_size, record["out_seq_size"])

            # Exclude solutions that do not match the observed output size
            prev_n_hyps = len(shape_params_hyps)
            shape_params_hyps = [
                params
                for idx, params in enumerate(shape_params_hyps)
                if in_to_out_sizes[idx, in_size] == record["out_seq_size"]
            ]
            logger.info(
                f"[In/out rel] Step {step}: rejected {prev_n_hyps - len(shape_params_hyps)}/{prev_n_hyps} hypotheses"
            )

            step += 1

        return self.in_out_rel_params

    def _verify_hypothesis_kernels_against_record(
        self,
        hypothesis: _SliHypothesis,
        in_seq_size: int,
        in_nan_range: Tuple[int, int] | None,
        out_seq_size: int,
        out_nan_idx: np.ndarray,
        out_nan_range: Tuple[int, int] | None,
    ):
        # All of our observations rely on the nan trick.
        if in_nan_range is None or out_seq_size == 0:
            return True

        # Reject if the nan trick's output is not compatible with the hypothesis
        new_kernels = determine_kernel_sparsity(
            hypothesis.params, *hypothesis.kernels, in_seq_size, in_nan_range, out_nan_idx
        )
        if new_kernels[0] is None:
            return False
        # Update the hypothesis in place
        hypothesis.kernels = new_kernels

        return True

    def _iter_nan_trick_params_for_hypothesis(self, params: SlidingWindowParams):
        # As specified in the sampler, for any given set of parameters, picking a nan range larger than the input
        # kernel size and ensuring that the pre-nan out size is larger than the output kernel size will let us
        # know with certainty whether the parameters' delays are matching the transform.
        min_nan_in_size = params.kernel_size_in
        # TODO!! more constraints, based on the sampler's edge cases
        # TODO! could we base constraints on the strides rather than the kernel sizes
        target_pre_nan_out_size = max(params.kernel_size_out, get_output_delay_bounds(params)[1])
        min_non_nan_in_size = max(
            params.get_min_input_size_for_out_size(target_pre_nan_out_size), params.kernel_size_in
        )

        # We'll start by going through the nan trick history. If we already have a nan trick record that validated
        # a phase for these parameters, we can skip testing that phase again
        nan_start_phases, nan_end_phases = set(range(params.stride_in)), set(range(params.stride_in))
        for record in self.nan_trick_history:
            if (
                record["in_nan_range"]
                and record["out_nan_range"]
                # TODO: constraints on out size or no?
                and record["in_nan_range"][0] >= min_non_nan_in_size
                and record["in_nan_range"][1] - record["in_nan_range"][0] >= min_nan_in_size
                and record["in_seq_size"] - record["in_nan_range"][1] >= min_non_nan_in_size
            ):
                nan_start_phases.discard(record["in_nan_range"][0] % params.stride_in)
                nan_end_phases.discard(record["in_nan_range"][1] % params.stride_in)

        # TODO
        size_factor = 3
        for nan_start_phase, nan_end_phase in zip_longest(nan_start_phases, nan_end_phases, fillvalue=0):
            # Align the nan start on the given phase while ensuring the pre-nan in size is large enough
            pre_nan_in_size = min_non_nan_in_size * size_factor
            pre_nan_in_size = pre_nan_in_size + ((nan_start_phase - pre_nan_in_size) % params.stride_in)

            # Then the nan end, ensuring the nan range is large enough
            post_nan_in_size = pre_nan_in_size + min_nan_in_size * size_factor
            post_nan_in_size = post_nan_in_size + ((nan_end_phase - post_nan_in_size) % params.stride_in)

            # The post-nan segment must also be large enough, but doesn't need to be phase aligned
            full_in_size = post_nan_in_size + min_non_nan_in_size * size_factor

            yield (full_in_size, (pre_nan_in_size, post_nan_in_size))

    def _debug_check_ref_params(
        self,
        sampler,
        event: str,
        other_params: SlidingWindowParams | None = None,
        allow_rejection: bool = False,
    ):
        """
        Debugging method for checking why a good reference hypothesis gets rejected.
        """
        # FIXME! review

        if self.debug_ref_params and (violations := sampler.get_violations(self.debug_ref_params)):
            violations_str = "\n\n-------------------\n\t".join(str(v) for v in violations)
            logger.info(
                f"{colors.RED}Reference hypothesis {_compare_sli_params_str(self.debug_ref_params)} "
                f"\nbecame incompatible with "
                f"the sampler after {event}"
                f"{_compare_sli_params_str(other_params, self.debug_ref_params) if other_params else ''}\n"
                f"{colors.YELLOW}Violations:\n\t{violations_str}{colors.RESET}"
            )

            if other_params:
                in_size = max(50, other_params.get_min_input_size_for_num_wins(10))
                in_seq = self.input_provider(in_size)
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
                logger.info(f"--> {colors.BLUE}Rejection is allowed{colors.RESET}")
                self.debug_ref_params = None
            else:
                raise RuntimeError()

    def _sli_search_integrate_nan_trick_record(self, sampler: SlidingWindowParamsSampler, record: dict):
        # Update our sampler with new constraints
        sampler.add_in_out_range_map(
            record["in_seq_size"], record["out_seq_size"], record["in_nan_range"], record["out_nan_range"]
        )
        self._debug_check_ref_params(sampler, "adding nan trick record")

    # TODO (major): split further into two steps: one for streaming params (out delay + ctx) using stride based
    # constraints, and a last step for kernel sizes by embedding the kernel sparsity solver
    def find_sliding_window_params(self):
        # Start by determining the input/output size relationship, it will heavily simplify the param search to
        # know it in advance
        in_out_rel_params = self.find_in_out_rel_params()
        sampler = SlidingWindowParamsSampler(*in_out_rel_params)

        # The NaN tricks we ran for the in/out size relation are relevant, we'll integrate them into the sampler
        for record in self.nan_trick_history:
            self._sli_search_integrate_nan_trick_record(sampler, record)

        step = 1
        out_sols = []
        while len(out_sols) < self.max_equivalent_sols:
            # Sample new sliding window parameters
            params = sampler.get_new_solution(same_family_as=(out_sols[0] if out_sols else None))
            if params is None:
                break

            hypothesis = _SliHypothesis(params, id=step)
            logger.info(
                f"[Sli params] Step {step}: {_compare_sli_params_str(hypothesis.params, self.debug_ref_params)}"
            )

            checks_passed = all(
                self._verify_hypothesis_kernels_against_record(hypothesis, **record)
                for record in self.nan_trick_history
            )
            if not checks_passed:
                # We don't break here - despite failing the kernel checks, we want to get at least one nan trick run
                # for this hypothesis.
                logger.info(f"{colors.RED}Hypothesis #{hypothesis.id} REJECTED after kernel check{colors.RESET}")

            for nan_trick_params in self._iter_nan_trick_params_for_hypothesis(hypothesis.params):
                record = self.run_nan_trick(*nan_trick_params)
                self._sli_search_integrate_nan_trick_record(sampler, record)

                if checks_passed and not self._verify_hypothesis_kernels_against_record(hypothesis, **record):
                    logger.info(f"{colors.RED}Hypothesis #{hypothesis.id} REJECTED after kernel check{colors.RESET}")
                    checks_passed = False
                if checks_passed and not sampler.is_compatible(hypothesis.params):
                    logger.info(f"{colors.RED}Hypothesis #{hypothesis.id} REJECTED by constraints{colors.RESET}")
                    checks_passed = False

                if not checks_passed:
                    break

            if checks_passed:
                out_sols.append(hypothesis.params)

            step += 1

        return out_sols


# TODO: allow transforms with multiple sequential inputs
#   -> Or simply call the function multiple times? unsure
@torch.no_grad()
def find_sliding_window_params(
    trsfm: Callable,
    input_provider: Callable[[int], Sequence] | SeqSpec,
    out_spec: SeqSpec | None = None,
    init_seq_size: int = 30,
    max_in_seq_size: int = 10_000,
    atol: float = 1e-5,
    max_equivalent_sols: int = 1,
    zero_size_exception_types: Tuple[type[Exception], ...] = (RuntimeError,),
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
        max_equivalent_sols=max_equivalent_sols,
        zero_size_exception_types=zero_size_exception_types,
        debug_ref_params=debug_ref_params,
    ).find_sliding_window_params()
