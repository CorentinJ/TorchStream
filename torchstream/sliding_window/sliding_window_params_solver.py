import logging
from functools import partial
from itertools import zip_longest
from typing import Callable, Iterable, List, Tuple

import torch
from colorama import Fore as colors
from opentelemetry import trace

from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS, ExceptionWithSubstring
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.stream_buffer import StreamBuffer
from torchstream.sliding_window.kernel_sparsity import KernelSparsitySampler
from torchstream.sliding_window.nan_trick import get_nan_idx
from torchstream.sliding_window.sliding_window_in_out_rel_sampler import (
    SlidingWindowInOutRelSampler,
)
from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
    get_canonicalized_min_in_size,
    get_output_delay_bounds,
    in_out_rel_repr,
)
from torchstream.sliding_window.sliding_window_params_sampler import SlidingWindowParamsSampler

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


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
        ref_size_rel = (real_params.canonicalized_in_out_shape_params) + (real_params.min_input_size,)
        ref_delays = real_params.output_delays
        ref_ctx = (real_params.streaming_context_size,)
    else:
        ref_params, ref_size_rel, ref_delays, ref_ctx = None, None, None, None

    params_size_rel = params.canonicalized_in_out_shape_params + (params.min_input_size,)
    return (
        f"\n\tparameters ({_compare_params_str(params.as_tuple(with_min_in_size=False), ref_params, 'ki,si,lp,rp,ko,so,lt,rt'.split(','))})"
        f"\n\twith shape ({_compare_params_str(params_size_rel, ref_size_rel, 's_i,s_o,isbc,osbc,mis'.split(','))})"
        f"\n\twith output delays ({_compare_params_str(params.output_delays, ref_delays)})"
        f"\n\twith context size {_compare_params_str((params.streaming_context_size,), ref_ctx)}"
    )


class _SliHypothesis:
    def __init__(self, params: SlidingWindowParams, id: int):
        self.params = params
        self.id = id
        self.kernel_sparsity_sampler = KernelSparsitySampler(params)


class SlidingWindowParamsSolver:
    def __init__(
        self,
        trsfm: Callable,
        input_provider: Callable[[int], StreamBuffer] | SeqSpec,
        out_spec: SeqSpec | None = None,
        init_seq_size: int = 30,
        max_in_seq_size: int = 10_000,
        atol: float = 1e-5,
        max_equivalent_sols: int = 1,
        zero_size_exception_signatures: Iterable[Exception | ExceptionWithSubstring] = DEFAULT_ZERO_SIZE_EXCEPTIONS,
        debug_ref_params: SlidingWindowParams | None = None,
    ):
        if isinstance(input_provider, SeqSpec):
            in_spec = input_provider
            input_provider = partial(StreamBuffer.randn, in_spec)

        self._trsfm = trsfm
        self.input_provider = input_provider
        self.out_spec = out_spec
        self.init_seq_size = init_seq_size
        self.max_in_seq_size = max_in_seq_size
        self.atol = atol
        self.max_equivalent_sols = max_equivalent_sols
        self.zero_size_exception_signatures = zero_size_exception_signatures

        self.in_out_rel_params = None
        self.min_in_size_bounds = [1, max_in_seq_size]
        self.nan_trick_history = []

        # FIXME!
        self.in_spec = self.input_provider(0).spec

        self.debug_ref_params = debug_ref_params
        if debug_ref_params:
            logger.info(f"Debug reference parameters: {_compare_sli_params_str(debug_ref_params)}")

    @property
    def step(self) -> int:
        """
        Solver step = number of times the transform has been
        """
        return len(self.nan_trick_history)

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
        if not isinstance(in_seq, StreamBuffer):
            raise TypeError(
                f"The input_provider function {self.input_provider} returned a {type(in_seq)} "
                f"when a Sequence was expected"
            )

        # Corrupt the given range of the input sequence with NaNs
        if in_nan_range:
            in_seq[slice(*in_nan_range)] = float("nan")

        out_seq = StreamBuffer.apply(
            self._trsfm, in_seq, self.out_spec, zero_size_exception_signatures=self.zero_size_exception_signatures
        )

        # Keep track of the outcome in the history
        out_nan_idx = get_nan_idx(out_seq)
        out_nan_range = (int(out_nan_idx[0]), int(out_nan_idx[-1] + 1)) if len(out_nan_idx) else None
        logger.info(f"Forwarded size {in_seq.size}->{out_seq.size} with nans {in_nan_range}->{out_nan_range}")
        record = {
            "in_seq_size": in_seq.size,
            "in_nan_range": in_nan_range,
            "out_seq_size": out_seq.size,
            "out_nan_idx": out_nan_idx,
            "out_nan_range": out_nan_range,
        }
        self.nan_trick_history.append(record)

        # Update our min input size bounds
        if out_seq.size > 0:
            self.min_in_size_bounds[1] = min(self.min_in_size_bounds[1], in_seq.size)
        else:
            self.min_in_size_bounds[0] = max(self.min_in_size_bounds[0], in_seq.size + 1)

        # Raise if we get no output with the maximum input size
        if in_seq_size == self.max_in_seq_size and out_seq.size == 0:
            raise RuntimeError(
                f"Your transform gave an output of size 0 given the maximum input size (={self.max_in_seq_size}). "
                f"Aborting.\n"
                f"It's possible you have specified a too broad exception for the zero_size_exception_signatures "
                f"argument."
            )

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

    @tracer.start_as_current_span("find_initial_input")
    def run_initial_input(self) -> dict:
        # TODO! doc
        # In the first part of the process, we'll forward inputs to the transform and stop as soon as we get a
        # output sequence of non-zero size
        while not any(record["out_seq_size"] for record in self.nan_trick_history):
            # Use sane defaults for the NaN trick
            record = self.run_nan_trick(self.init_seq_size, (self.init_seq_size // 2, self.init_seq_size // 2 + 1))
            if record["out_seq_size"]:
                break

            # As long as we haven't had a valid output, we'll increase the input size. We do this before involving
            # the sampler, otherwise we may be stuck sampling for a while before getting decent candidates.
            self.init_seq_size = min(10 * self.init_seq_size, self.max_in_seq_size)
            logger.info(
                f"[Init input] Step {self.step} - Transform failed with input size {record['in_seq_size']}. "
                f"Increasing init sequence size to {self.init_seq_size}"
            )

        return self.nan_trick_history[0]

    @tracer.start_as_current_span("find_in_out_rel_params")
    def find_in_out_size_params(self) -> Tuple[int, int, int, int]:
        # TODO! doc
        if self.in_out_rel_params:
            return self.in_out_rel_params

        # Ensure we have at least one example input before starting
        self.run_initial_input()

        # Integrate the history from the initial input runs in the solver
        sampler = SlidingWindowInOutRelSampler()
        for record in self.nan_trick_history:
            sampler.add_in_out_size(record["in_seq_size"], record["out_seq_size"])

        while True:
            shape_params, next_in_size = sampler.solve(self.min_in_size_bounds[0], self.max_in_seq_size)
            if shape_params:
                # Params uniquely determined, let's update our state and our lower bound for the input size based
                # on them
                self.in_out_rel_params = shape_params
                self.min_in_size_bounds[0] = max(
                    self.min_in_size_bounds[0], get_canonicalized_min_in_size(*shape_params)
                )
                logger.info(
                    f"[In/out rel] Step {self.step} - Converged to in/out size relation:"
                    f"\n\t{in_out_rel_repr(*shape_params)}"
                )

                return self.in_out_rel_params
            else:
                logger.info(f"[In/out rel] Step {self.step}")

            # If we have no solution, the transform is not a sliding window.
            if not next_in_size:
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

            # TODO? should we try different nan idx values here already?
            #   -> Yes! That would help with converging towards solutions faster. Determine a heuristic size based
            #      on the input size relations
            nan_idx = (next_in_size // 2, next_in_size // 2 + 1)
            record = self.run_nan_trick(next_in_size, nan_idx)
            sampler.add_in_out_size(next_in_size, record["out_seq_size"])

    @tracer.start_as_current_span("find_min_input_size")
    def find_min_input_size(self) -> int:
        # TODO! doc
        # Ensure we have at least one example input before starting
        self.run_initial_input()

        while self.min_in_size_bounds[0] < self.min_in_size_bounds[1]:
            # Heuristic: if the canonical min input size hasn't been tested, we'll test it. Most often that will be
            # the actual minimum input size. Otherwise we'll bisect
            canon_min_in_size = None
            if self.in_out_rel_params is not None:
                canon_min_in_size = get_canonicalized_min_in_size(*self.in_out_rel_params)
            if canon_min_in_size is not None and not any(
                record["in_seq_size"] == canon_min_in_size for record in self.nan_trick_history
            ):
                in_size = canon_min_in_size
            else:
                lower_bound = max(
                    (record["in_seq_size"] for record in self.nan_trick_history if record["out_seq_size"] == 0),
                    default=canon_min_in_size or 1,
                )
                upper_bound = min(
                    (record["in_seq_size"] for record in self.nan_trick_history if record["out_seq_size"] > 0),
                    default=self.max_in_seq_size + 1,
                )
                in_size = (lower_bound + upper_bound) // 2

            # TODO? should we try different nan idx values here already?
            #   -> Yes! That would help with converging towards solutions faster. Determine a heuristic size based
            #      on the input size relations
            nan_idx = (in_size // 2, in_size // 2 + 1)
            self.run_nan_trick(in_size, nan_idx)

            if self.min_in_size_bounds[0] < self.min_in_size_bounds[1]:
                range_str = f"range {self.min_in_size_bounds}"
            else:
                range_str = f"is {self.min_in_size_bounds[0]}"
            logger.info(f"[Min input size] Step {self.step} - min in size {range_str}")

        return self.min_in_size_bounds[0]

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
    ):
        """
        Debugging method for checking why a good reference hypothesis gets rejected.
        """
        if self.debug_ref_params and (violations := sampler.get_violations(self.debug_ref_params)):
            violations_str = "\n\n-------------------\n\t".join(str(v) for v in violations)
            logger.info(
                f"{colors.RED}Reference hypothesis {_compare_sli_params_str(self.debug_ref_params)} "
                f"\nbecame incompatible with "
                f"the sampler after {event}"
                f"{_compare_sli_params_str(other_params, self.debug_ref_params) if other_params else ''}\n"
                f"{colors.YELLOW}Violations:\n\t{violations_str}{colors.RESET}"
            )

    def _sli_search_integrate_nan_trick_record(
        self, sampler: SlidingWindowParamsSampler, hypotheses: List[_SliHypothesis], record: dict
    ) -> List[_SliHypothesis]:
        out_hyps = []

        sampler.add_in_out_range_map(
            record["in_seq_size"], record["out_seq_size"], record["in_nan_range"], record["out_nan_range"]
        )
        for hypothesis in hypotheses:
            self._debug_check_ref_params(sampler, "adding nan trick record", hypothesis.params)
            if not sampler.is_compatible(hypothesis.params):
                logger.info(f"Hypothesis #{hypothesis.id} REJECTED by constraints")
                continue

            if record["in_nan_range"] and record["out_seq_size"]:
                hypothesis.kernel_sparsity_sampler.add_in_out_map(
                    record["in_seq_size"], record["in_nan_range"], record["out_nan_idx"]
                )
                if not hypothesis.kernel_sparsity_sampler.has_solution():
                    logger.info(f"Hypothesis #{hypothesis.id} REJECTED after kernel check")
                    continue

            out_hyps.append(hypothesis)

        return out_hyps

    # TODO (major): split further into two steps: one for streaming params (out delay + ctx) using stride based
    # constraints, and a last step for kernel sizes by embedding the kernel sparsity solver
    def find_sliding_window_params(self):
        # Start by determining the input/output size relationship, it will heavily simplify the param search to
        # know it in advance
        in_out_rel_params = self.find_in_out_size_params()
        min_input_size = self.find_min_input_size()
        sampler = SlidingWindowParamsSampler(*in_out_rel_params, min_input_size)

        # The NaN tricks we ran for the in/out size relation are relevant, we'll integrate them into the sampler
        for record in self.nan_trick_history:
            self._sli_search_integrate_nan_trick_record(sampler, [], record)

        n_hyps = 0
        out_sols = []
        while len(out_sols) < self.max_equivalent_sols:
            # Sample new sliding window parameters
            params = sampler.get_new_solution(same_family_as=(out_sols[0].params if out_sols else None))
            if params is None:
                break
            n_hyps += 1

            hypothesis = _SliHypothesis(params, id=n_hyps)
            logger.info(
                f"[Sli params] Step {self.step} - Testing hypothesis #{hypothesis.id}:"
                + _compare_sli_params_str(hypothesis.params, self.debug_ref_params)
            )

            for record in self.nan_trick_history:
                if record["in_nan_range"] and record["out_seq_size"]:
                    hypothesis.kernel_sparsity_sampler.add_in_out_map(
                        record["in_seq_size"], record["in_nan_range"], record["out_nan_idx"]
                    )
            checks_passed = hypothesis.kernel_sparsity_sampler.has_solution()
            if not checks_passed:
                # We don't break here - despite failing the kernel checks, we want to get at least one nan trick run
                # for this hypothesis. This will guide the sampler towards better hypotheses next step
                logger.info(f"Hypothesis #{hypothesis.id} REJECTED after kernel check")
            else:
                out_sols.append(hypothesis)

            for nan_trick_params in self._iter_nan_trick_params_for_hypothesis(hypothesis.params):
                record = self.run_nan_trick(*nan_trick_params)
                out_sols = self._sli_search_integrate_nan_trick_record(sampler, out_sols, record)
                checks_passed &= hypothesis in out_sols

                if not checks_passed:
                    break

            if checks_passed:
                logger.info(f"Hypothesis #{hypothesis.id} ACCEPTED as solution - all checks passed")

        return [hyp.params for hyp in out_sols]


@torch.no_grad()
def find_sliding_window_params(
    trsfm: Callable,
    input_provider: Callable[[int], StreamBuffer] | SeqSpec,
    out_spec: SeqSpec | None = None,
    init_seq_size: int = 30,
    max_in_seq_size: int = 10_000,
    atol: float = 1e-5,
    max_equivalent_sols: int = 1,
    zero_size_exception_signatures: Iterable[Exception | ExceptionWithSubstring] = DEFAULT_ZERO_SIZE_EXCEPTIONS,
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

    :param input_spec: specification for the input format of the transform. The transform must accept the data format
    described in the input spec as positional arguments.
    :param output_spec: same as input_spec but for the output of the transform. If the transform has multiple
    sequential outputs, they must be returned as an iterable matching the output spec. If the output spec is
    identical to the input spec, it can be omitted, and the input spec will be used instead.
    :param input_provider: a function that takes an integer representing the sequence size, and returns a Sequence of
    this size.
    TODO!: rewrite docs
    """
    return SlidingWindowParamsSolver(
        trsfm=trsfm,
        input_provider=input_provider,
        out_spec=out_spec,
        init_seq_size=init_seq_size,
        max_in_seq_size=max_in_seq_size,
        atol=atol,
        max_equivalent_sols=max_equivalent_sols,
        zero_size_exception_signatures=zero_size_exception_signatures,
        debug_ref_params=debug_ref_params,
    ).find_sliding_window_params()
