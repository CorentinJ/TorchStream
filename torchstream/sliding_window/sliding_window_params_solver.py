import logging
import math
from itertools import zip_longest
from typing import Callable, Iterable, List, Tuple

from colorama import Fore as colors
from opentelemetry import trace

from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS, ExceptionWithSubstring
from torchstream.sequence.sequence import SeqSpec
from torchstream.sliding_window.kernel_sparsity import KernelSparsitySampler
from torchstream.sliding_window.nan_trick import run_nan_trick
from torchstream.sliding_window.sliding_window_in_out_size_sampler import (
    SlidingWindowInOutSizeSampler,
)
from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
    get_canonical_min_in_size,
    get_output_delay_bounds,
    in_out_size_rel_repr,
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
        ref_size_rel = (real_params.canonical_in_out_size_params) + (real_params.min_input_size,)
        ref_delays = real_params.output_delays
        ref_ctx = (real_params.streaming_context_size,)
    else:
        ref_params, ref_size_rel, ref_delays, ref_ctx = None, None, None, None

    params_size_rel = params.canonical_in_out_size_params + (params.min_input_size,)
    return (
        f"\n\tparameters ({_compare_params_str(params.as_tuple(with_min_in_size=False), ref_params, 'ki,si,lp,rp,ko,so,lt,rt'.split(','))})"
        f"\n\twith in/out size relation ({_compare_params_str(params_size_rel, ref_size_rel, 's_i,s_o,isbc,osbc,mis'.split(','))})"
        f"\n\twith output delays ({_compare_params_str(params.output_delays, ref_delays)})"
        f"\n\twith context size {_compare_params_str((params.streaming_context_size,), ref_ctx)}"
    )


class MaximumSequenceSizeReachedError(RuntimeError):
    pass


class _SliHypothesis:
    def __init__(self, params: SlidingWindowParams, id: int):
        self.params = params
        self.id = id
        self.kernel_sparsity_sampler = KernelSparsitySampler(params)


class SlidingWindowParamsSolver:
    """
    This class takes a sequence-to-sequence transform (neural net, signal processing function, time series analysis
    function, ...) and a specification of its input and output formats. It then generates inputs of varying sizes
    with NaNs in specific locations, analyzing the outputs to determine whether the transform is a sliding window
    operation. If it is, it will find these sliding window parameters (or functionally equivalent ones (*)) and return
    them.

    Knowing the sliding window parameters of a transform allows for deriving a precise and exact input to output
    mapping, which in turn makes the transform trivially streamable without approximations (see SlidingWindowStream).
    It also allows for deriving core streaming metrics: time to first output, latency, context size, optimal chunk
    size, ...

    The majority of sequence-to-sequence neural networks are, fully or in part (**), acting as a sliding window
    operation. Convolutions, padding, pooling, upsampling, element-wise operations, and many other layers are
    cases of sliding window operations, and chained together they act as a single sliding window transform (***) with
    compound parameters. Leveraging this property allows streaming even complex architectures exactly without
    resorting to a challenging implementation. Determining these compound parameters manually is however tedious
    and error-prone, hence the need for this solver.

    The transform must meet some constraints:
    - It must accept NaNs as input and appropriately propagate them to output elements that depend on them. The vast
    majority of pure python, numpy, scipy and torch operations behave this way. If using a third-party library that
    raises an error on NaNs (e.g. librosa), consider patching the underlying functions using torchstream's
    intercept_calls context manager for the duration of the solver run. Only the nan input and output positions matter
    to the solver, not the values of the other elements.
    - It must take and output sequential data only. If there are multiple inputs, they must share a time resolution
    expressible as a ratio of integers. The same applies to multiple outputs. If your transform has an input that
    is constant for the duration of the sequence, provide instead a wrapper that takes only the sequential input and
    injects the constant input internally. If your transform has an input that changes based on the sequence position,
    it is sequential data by definition. Consider passing it as such or making it internal to the transform.
    - Evidently, it must behave as a sliding window operations. It is frequent for neural networks not initially
    designed for streaming to feature sequence-global operation such as a mean or a norm over the sequence dimension.
    Such operations technically have an infinite input kernel size and cannot be streamed exactly. They may be
    approximated for streaming with success, but you will have to mitigate their NaN propagation effect for the
    duration of the solver run, e.g. using the intercept_calls context manager to patch them into identity functions.
    Autoregressive operations are not sliding window operations and will also need to be patched. In either case, the
    solver is able to detect these transforms and raise an appropriate error. Layers with a variable receptive field
    (e.g. some attention layers) are not modeled by this solver and will need an adhoc implementation. Layers that
    cannot operate without seeing the entire input sequence (e.g. most attention layers) are inherently not
    streamable without serious approximations nor do they behave as sliding window operations. If your model features
    such layers, unfortunately the only recourse might be to redesign and to retrain it.

    Note that the transform does NOT need to be deterministic for the solver to work, as it operates solely on the
    NaN propagation behaviour.

    This solver emits logging information at the INFO level, and tracing spans using OpenTelemetry. Ensure you have
    a logging handler setup at the INFO level or lower to analyze its progress.

    (*) Different sliding window parameters are considered to be functionally equivalent if they all produce the exact
    same input to output mapping. For instance, Conv1d(kernel_size=2, stride=1, padding=1) and
    ConvTranspose1d(kernel_size=2, stride=1) produce the exact same input to output mappings. They technically
    only differ by their kernel.

    (**) If your end goal is latency reduction by streaming your model's output when all the input is available,
    you can afford to run eventual non-streamable operations on the full input before streaming the rest of the model.

    (***) The solver models transforms with an input to output size relationship of the form:
    {
        output_size = ((input_size + a) // b) * c + d, if input_size >= e
        output_size = 0, otherwise
    }
    This covers the vast majority of composition of sliding window operations. For a neural network, on any layer
    with an input stride > 1, the current combined stride of the model must be expressible as either 1 / x or x / 1.
    A couple of examples:
      - Layer 11: transposed conv with output stride = 3, Layer 2: conv with input stride = 6
          -> after Layer 1 our combined stride is 3 (3 / 1 -> OK) and after Layer 2 it's 3 / 6 = 1 / 2 -> OK.
      - Layer 1: conv with input stride = 2, Layer 2: conv with input stride = 3
          -> after Layer 1 our combined stride is 1 / 2, and after Layer 2 it's 1 / 6. All OK
      - Layer 1: transposed conv with output stride = 3, Layer 2: conv with input stride = 2
          -> after Layer 1 our combined stride is 3 (3 / 1 -> OK) and after Layer 2 it's 3 / 2 -> NOT OK. The solver
          will fail.
      - Layer 1: conv with input stride = 2, Layer 2: transposed conv with output stride = 3
          -> After Layer 1 our combined stride is 1 / 2, and after Layer 2 it's 3 / 2 BUT the check only needs to hold
          on layers with an input stride > 1. E.g. adding another conv with input stride = 2 as Layer 3 will fail
          the solver.
    Note that in practice, models will almost always meet this requirement. Indeed, most models either only upsample,
    downsample, or downsample first before upsampling. Only in the case where a model upsamples before downsampling
    could we have this issue (provided the strides do not meet the condition) - and I don't know yet of such a model.

    :param trsfm: a callable transform taking sequential data as input and outputting sequential data.
    :param in_spec: specification for the input format of the transform, as positional arguments.
    :param out_spec: specification for the output format of the transform. If omitted, assumed to be identical to the
    input spec.
    :param init_seq_size: initial input sequence size to try when probing the transform. You can increase this if your
    model has a high minimum input size, but the solver should quickly converge to it anyway.
    :param max_in_out_seq_size: maximum input and output sequence size to try when probing the transform. If the solver
    reaches this size without being able to determine the sliding window parameters, it will assume there is no
    solution and raise a RuntimeError.
    :param zero_size_exception_signatures: an iterable of exception types or tuples (exception_type, substring) that
    indicate that the transform cannot produce an output because the input size is too small.
    :param debug_ref_params: if provided, the solver will compare its hypotheses to these reference parameters in the
    logs and will raise an exception if these parameters become incompatible with the solver.
    """

    def __init__(
        self,
        trsfm: Callable,
        in_spec: SeqSpec,
        out_spec: SeqSpec | None = None,
        init_seq_size: int = 30,
        max_in_out_seq_size: int = 100_000,
        zero_size_exception_signatures: Iterable[Exception | ExceptionWithSubstring] = DEFAULT_ZERO_SIZE_EXCEPTIONS,
        debug_ref_params: SlidingWindowParams | None = None,
    ):
        self._trsfm = trsfm
        self.in_spec = in_spec
        self.out_spec = out_spec or in_spec
        self.init_seq_size = init_seq_size
        self.max_in_out_seq_size = max_in_out_seq_size
        self.zero_size_exception_signatures = zero_size_exception_signatures

        self._in_out_size_params = None
        self._min_in_size_bounds = [1, max_in_out_seq_size]
        self.nan_trick_history = []

        self.debug_ref_params = debug_ref_params
        if debug_ref_params:
            logger.info(f"Debug reference parameters: {_compare_sli_params_str(debug_ref_params)}")

    @property
    def step(self) -> int:
        """
        Solver step = number of times the transform has been run
        """
        return len(self.nan_trick_history)

    def run_nan_trick(
        self, in_seq_size: int, in_nan_range: Tuple[int, int] | None, raise_on_max_seq_size: bool = True
    ) -> dict:
        """
        Forwards an input of size `in_seq_size` with NaNs in the range `in_nan_range` through the transform, storing
        the inputs and outputs NaN positions. The function will raise if the transform does not behave as expected.
        """
        if in_nan_range and not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq_size):
            raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")

        # Running the same nan trick twice is a waste of compute. Callers are expected not to do this.
        assert not any(
            (in_seq_size, in_nan_range) == (record["in_seq_size"], record["in_nan_range"])
            for record in self.nan_trick_history
        ), "Internal error: reusing previously seen NaN trick parameters"

        # Get an input of said size and perform the nan trick on the actual transform
        in_seq = self.in_spec.new_randn_sequence(in_seq_size)
        out_seq, out_nan_idx = run_nan_trick(
            self._trsfm, in_seq, in_nan_range, self.out_spec, self.zero_size_exception_signatures
        )

        # Keep track of the outcome in the history
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
            self._min_in_size_bounds[1] = min(self._min_in_size_bounds[1], in_seq.size)
        else:
            self._min_in_size_bounds[0] = max(self._min_in_size_bounds[0], in_seq.size + 1)

        if raise_on_max_seq_size and max(in_seq_size, out_seq.size) >= self.max_in_out_seq_size:
            raise MaximumSequenceSizeReachedError(
                f"Reached maximum input/output sequence size ({self.max_in_out_seq_size:,}) "
                f"with a {in_seq_size:,} -> {out_seq.size:,} forward pass. Aborting.\n"
                f"If you believe a valid solution exists, consider increasing this limit."
            )

        return record

    @tracer.start_as_current_span("find_valid_input")
    def find_valid_input(self):
        """
        This function tries increasing input sizes until it finds one that produces:
        - A non zero output size
        - With NaNs propagated in its output - transforms that consistently miss an input range are not modeled
        by this solver (they are not considered useful)
        - Without NaNs as its first or last elements, in order to detect non-finite kernels early on in the solver.
        """
        if self.nan_trick_history:
            return

        # In the first part of the process, we'll forward inputs to the transform and stop as soon as we get a
        # output sequence of non-zero size
        while True:
            # Use sane defaults for the NaN trick
            nan_start = self.init_seq_size // 2 - self.init_seq_size // 20
            nan_end = self.init_seq_size // 2 + int(math.ceil(self.init_seq_size / 20))
            record = self.run_nan_trick(self.init_seq_size, (nan_start, nan_end), raise_on_max_seq_size=False)

            # Check whether the output is valid
            if record["out_seq_size"] == 0:
                fail_reason = "zero-sized output"
            elif record["out_nan_range"] is None:
                fail_reason = "no NaN propagation"
            else:
                if record["out_nan_range"] == (0, record["out_seq_size"]):
                    fail_reason = "full NaN output"
                elif record["out_nan_range"][0] == 0:
                    fail_reason = "first output is NaN"
                elif record["out_nan_range"][1] == record["out_seq_size"]:
                    fail_reason = "last output is NaN"
                else:
                    return record

            if record["out_seq_size"] >= self.max_in_out_seq_size or record["in_seq_size"] >= self.max_in_out_seq_size:
                break

            # As long as we haven't had a valid output, we'll increase the input size.
            self.init_seq_size = min(10 * self.init_seq_size, self.max_in_out_seq_size)
            logger.info(
                f"[Init input] Step {self.step} - Transform failed ({fail_reason}), "
                f"increasing init sequence size to {self.init_seq_size}"
            )

        # At this point we assume the transform is not suitable, give a helpful error message to the user
        if fail_reason == "zero-sized output":
            raise RuntimeError(
                f"Your transform gave an output of size 0 given the maximum input size (={self.max_in_out_seq_size}). "
                f"Aborting.\n"
                f"It's possible you have specified a too broad exception for the zero_size_exception_signatures "
                f"argument."
            )
        elif fail_reason == "no NaN propagation":
            raise RuntimeError("Your transform never propagated any NaNs from its input.")
        elif fail_reason == "full NaN output":
            raise RuntimeError(
                f"Your transform outputs NaNs covering the entire output (in_size={record['in_seq_size']}, "
                f"out_size={record['out_seq_size']}). This likely means that an operation in your transform "
                f"broadcasts an input element to all output elements, like a mean, batchnorm, etc... One solution "
                f"might be to patch any such operation using torchstream's intercept_calls context manager to be an "
                f"identity function for the duration of the solver run, and approximate it later for streaming."
            )
        elif fail_reason == "first output is NaN":
            raise RuntimeError(
                f"Your transform outputs NaNs at the start of the output sequence (in_size={record['in_seq_size']}, "
                f"out_size={record['out_seq_size']}, out_nan_range={record['out_nan_range']}). Check your model for "
                f"an operation that behaves this way. If this is a critical operation, your model may not be "
                f"streamable."
            )
        else:
            raise RuntimeError(
                f"Your transform outputs NaNs at the end of the output sequence (in_size={record['in_seq_size']}, "
                f"out_size={record['out_seq_size']}, out_nan_range={record['out_nan_range']}). This likely means that "
                f"you have an autoregressive operation in your model (e.g. LSTM, cumsum, ...) that keeps producing "
                f"NaNs onces it has seen one. These operations are usually trivially streamable, but you'll need to "
                f"prevent their NaN propagation for the duration of the solver run, e.g. by patching them into "
                f"identity functions using torchstream's intercept_calls context manager."
            )

    @tracer.start_as_current_span("find_in_out_size_params")
    def find_in_out_size_params(self) -> Tuple[int, int, int, int]:
        # TODO! doc
        if self._in_out_size_params:
            return self._in_out_size_params

        # Ensure we have at least one example input before starting
        self.find_valid_input()

        # Integrate the history from the initial input runs in the solver
        sampler = SlidingWindowInOutSizeSampler()
        for record in self.nan_trick_history:
            sampler.add_in_out_size(record["in_seq_size"], record["out_seq_size"])

        while True:
            max_in_size = min(
                10 * max((record["in_seq_size"] for record in self.nan_trick_history)),
                self.max_in_out_seq_size,
            )
            size_params, next_in_size = sampler.solve(self._min_in_size_bounds[0], max_in_size)
            if size_params:
                # Params uniquely determined, let's update our state and our lower bound for the input size based
                # on them
                self._in_out_size_params = size_params
                self._min_in_size_bounds[0] = max(self._min_in_size_bounds[0], get_canonical_min_in_size(*size_params))
                logger.info(
                    f"[In/out size rel] Step {self.step} - Converged to in/out size relation:"
                    f"\n\t{in_out_size_rel_repr(*size_params)}"
                )

                return self._in_out_size_params
            else:
                logger.info(f"[In/out size rel] Step {self.step}")

            # If we have no solution, the transform is not a sliding window.
            if not next_in_size:
                raise RuntimeError(
                    "Could not determine input/output size relationship for your transform. This likely means that "
                    "your transform does not behave like a sliding window with a kernel of fixed size."
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
        self.find_valid_input()

        while self._min_in_size_bounds[0] < self._min_in_size_bounds[1]:
            # Heuristic: if the canonical min input size hasn't been tested, we'll test it. Most often that will be
            # the actual minimum input size. Otherwise we'll bisect
            canon_min_in_size = None
            if self._in_out_size_params is not None:
                canon_min_in_size = get_canonical_min_in_size(*self._in_out_size_params)
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
                    default=self.max_in_out_seq_size + 1,
                )
                in_size = (lower_bound + upper_bound) // 2

            # TODO? should we try different nan idx values here already?
            #   -> Yes! That would help with converging towards solutions faster. Determine a heuristic size based
            #      on the input size relations
            nan_idx = (in_size // 2, in_size // 2 + 1)
            self.run_nan_trick(in_size, nan_idx)

            if self._min_in_size_bounds[0] < self._min_in_size_bounds[1]:
                range_str = f"range {self._min_in_size_bounds}"
            else:
                range_str = f"is {self._min_in_size_bounds[0]}"
            logger.info(f"[Min input size] Step {self.step} - min input size {range_str}")

        return self._min_in_size_bounds[0]

    def _iter_nan_trick_params_for_hypothesis(self, params: SlidingWindowParams, upsize_factor: int):
        if upsize_factor < 1:
            raise ValueError(f"Upsize factor must be >= 1, got {upsize_factor}")

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

        for nan_start_phase, nan_end_phase in zip_longest(nan_start_phases, nan_end_phases, fillvalue=0):
            # Align the nan start on the given phase while ensuring the pre-nan in size is large enough
            pre_nan_in_size = min_non_nan_in_size * upsize_factor
            pre_nan_in_size = pre_nan_in_size + ((nan_start_phase - pre_nan_in_size) % params.stride_in)

            # Then the nan end, ensuring the nan range is large enough
            post_nan_in_size = pre_nan_in_size + min_nan_in_size * upsize_factor
            post_nan_in_size = post_nan_in_size + ((nan_end_phase - post_nan_in_size) % params.stride_in)

            # The post-nan segment must also be large enough, but doesn't need to be phase aligned
            full_in_size = post_nan_in_size + min_non_nan_in_size * upsize_factor

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
    def find_sliding_window_params(
        self, max_equivalent_sols: int = 1, hyp_test_upsize_factor: int = 3, max_hypotheses: int = 300
    ) -> List[SlidingWindowParams]:
        """
        Performs the sliding window parameter search

        :param max_equivalent_sols: maximum number of functionally equivalent solutions to find before returning. All
        solutions will share the same stride, size relation, and the same optimal streaming parameters (output delays,
        context size). This parameter exists to explore the different kernel size, padding & trimming combinations that
        can yield the same input to output mapping. With a finite maximum number of solutions, there is no guarantee
        that the solver will return precisely the sliding window parameters of the given transform, only functionally
        equivalent ones.
        TODO: offer to fully determine the kernel sparsity
        :param hyp_test_upsize_factor: when a hypothesis is found, inputs are generated with sizes based on the
        hypothesis' parameters, and upscaled by this factor. In case the hypothesis is incorrect, the upscaling allows
        for pruning similar hypotheses with slightly higher parameters ahead of time, reducing the total number of
        steps necessary to converge. The tradeoff is that larger input sizes will take more time for the model to
        process.
        :param max_hypotheses: maximum number of hypotheses to consider before raising a RuntimeError. The majority
        of transforms should be solved before reaching 10 hypotheses, this is a limit for dealing with edge cases,
        namely transforms with a receptive field of varying size.

        :return: a list of SlidingWindowParams instances, each functionally equivalent to the transform's actual
        sliding window parameters.
        """
        # Start by determining the input/output size relationship, it will heavily simplify the param search to
        # know it in advance
        in_out_size_params = self.find_in_out_size_params()
        min_input_size = self.find_min_input_size()
        sampler = SlidingWindowParamsSampler(*in_out_size_params, min_input_size)

        # The NaN tricks we ran for the in/out size relation are relevant, we'll integrate them into the sampler
        for record in self.nan_trick_history:
            self._sli_search_integrate_nan_trick_record(sampler, [], record)

        n_hyps = 0
        out_sols = []
        while len(out_sols) < max_equivalent_sols:
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

            for nan_trick_params in self._iter_nan_trick_params_for_hypothesis(
                hypothesis.params, hyp_test_upsize_factor
            ):
                record = self.run_nan_trick(*nan_trick_params)
                out_sols = self._sli_search_integrate_nan_trick_record(sampler, out_sols, record)
                checks_passed &= hypothesis in out_sols

                if not checks_passed:
                    break

            if checks_passed:
                logger.info(f"Hypothesis #{hypothesis.id} ACCEPTED as solution - all checks passed")

            if n_hyps >= max_hypotheses:
                raise RuntimeError(
                    f"Reached maximum number of hypotheses ({max_hypotheses}) without converging to a solution. "
                    f"Aborting.\n"
                    f"This likely means your transform does not behave according to our sliding window model:\n"
                    f"  - it may have a receptive field of varying size\n"
                    f"  - it may break assumptions (stride <= kernel_size, padding < kernel_size, ...)\n"
                )

        return [hyp.params for hyp in out_sols]


def find_sliding_window_params(
    trsfm: Callable,
    in_spec: SeqSpec,
    out_spec: SeqSpec | None = None,
    init_seq_size: int = 30,
    max_in_out_seq_size: int = 100_000,
    max_equivalent_sols: int = 1,
    max_hypotheses: int = 300,
    hyp_test_upsize_factor: int = 3,
    zero_size_exception_signatures: Iterable[Exception | ExceptionWithSubstring] = DEFAULT_ZERO_SIZE_EXCEPTIONS,
    debug_ref_params: SlidingWindowParams | None = None,
) -> List[SlidingWindowParams]:
    """
    Convenience wrapper around SlidingWindowParamsSolver.find_sliding_window_params. Refer to the class constructor
    and method for parameter documentation.
    """
    return SlidingWindowParamsSolver(
        trsfm=trsfm,
        in_spec=in_spec,
        out_spec=out_spec,
        init_seq_size=init_seq_size,
        max_in_out_seq_size=max_in_out_seq_size,
        zero_size_exception_signatures=zero_size_exception_signatures,
        debug_ref_params=debug_ref_params,
    ).find_sliding_window_params(max_equivalent_sols, hyp_test_upsize_factor, max_hypotheses)
