import logging
import math
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch

from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS, ExceptionWithSubstring
from torchstream.sequence.sequence import SeqSpec, Sequence
from torchstream.sliding_window.kernel_sparsity import get_nan_map
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams, get_output_delay

logger = logging.getLogger(__name__)


def get_nan_idx(x: Sequence) -> np.ndarray:
    """
    Given a sequence, returns the indices along the sequence dimension where NaNs are found. NaNs are expected to be
    found along the same indices across all buffers of the sequence, otherwise the function raises a ValueError.
    """
    if not x.size:
        return np.empty(0, dtype=np.int64)

    for arr, seqdim, scale in zip(x.data, x.seq_dims, x.seq_scales):
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()

        arr_is_nan = np.isnan(arr)
        

    # TODO: is_nan -> any reduction instead
    # Use flatnonzero maybe?
    x = x.mean(axis=tuple(i for i in range(x.ndim) if i != axis))

    return np.where(np.isnan(x))[0]


def run_nan_trick(
    trsfm: Callable,
    in_seq: Sequence,
    in_nan_range: Tuple[int, int] | None,
    out_spec: Optional[SeqSpec] = None,
    zero_size_exception_signatures: Iterable[Exception | ExceptionWithSubstring] = DEFAULT_ZERO_SIZE_EXCEPTIONS,
) -> Tuple[Sequence, np.ndarray]:
    """
    TODO: doc
    """
    if not in_seq.size:
        raise ValueError(f"Input sequence size must be greater than 0, got {in_seq.size}")
    if in_nan_range and not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq.size):
        raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")
    out_spec = out_spec or in_seq.spec

    # Corrupt the given range of the input sequence with NaNs
    if in_nan_range:
        in_seq[slice(*in_nan_range)] = float("nan")

    # Forward the input through the transform
    out_seq = in_seq.apply(trsfm, out_spec, zero_size_exception_signatures=zero_size_exception_signatures)

    out_nan_idx = get_nan_idx(out_seq)
    logger.info(f"Got a {tuple(out_seq.shape)} shaped output with nans at {out_nan_idx}")

    return out_seq, out_nan_idx


def get_context_size_empirically(
    params: SlidingWindowParams,
    trsfm: Callable | None = None,
    in_spec: SeqSpec | None = None,
    out_spec: SeqSpec | None = None,
) -> int:
    if trsfm:
        assert in_spec is not None, "Both trsfm and in_spec must be provided if either is provided"
    else:
        assert in_spec is None and out_spec is None

    # Heuristic for the input size, asserts will catch if it's not enough
    base_in_size = (params.streaming_context_size + params.get_min_input_size_for_out_size(10)) * 3

    phase_ctxs = []
    for phase_offset in range(params.stride_in):
        # We'll mimick the streaming. We begin by saying that the stream has already performed a step with the given
        # input size, and see where the output position of the stream is.
        in_size = base_in_size + phase_offset
        *_, out_size = params.get_metrics_for_input(in_size)
        out_delay = get_output_delay(params, in_size)
        stream_out_pos = out_size - out_delay
        assert stream_out_pos > 0, "Input size is not sufficient"

        # Then we perform a second step with the same input size (it's a detail: as long as it produces enough
        # outputs to measure the context we're good).
        # We operate as if the first input window is from the previous step, and we ensure it sees NaNs. This way
        # we can see how far the nans from "the previous step" propagate in the output of the current step.
        if trsfm is None:
            out_nan_map = get_nan_map(params, in_size, in_nan_range=(0, params.stride_in))
            out_nan_idx = np.where(out_nan_map == 2)[0]
        else:
            out_seq, out_nan_idx = run_nan_trick(
                trsfm,
                in_seq=in_spec.new_randn_sequence(in_size),
                in_nan_range=(0, params.stride_in),
                out_spec=out_spec,
            )
            assert out_seq.size == out_size, "Transform doesn't match the given sliding window params"
        last_nan_idx = out_nan_idx[-1] if len(out_nan_idx) else -1

        # Do the maths to figure out the context size for this phase
        wins_to_drop = min(
            int(math.ceil((stream_out_pos - last_nan_idx) / params.stride_out)),
            stream_out_pos // params.stride_out,
        )
        base_wins_to_drop = in_size // params.stride_in
        wins_to_keep = base_wins_to_drop - wins_to_drop

        ctx = max(0, (wins_to_keep - 1) * params.stride_in + (in_size % params.stride_in) + 1)
        assert (in_size - ctx) // params.stride_in == base_wins_to_drop - wins_to_keep
        phase_ctxs.append(ctx)

    # The final context is the max of all phase dependant contexts
    ctx = max(phase_ctxs)
    assert min(phase_ctxs) + params.stride_in > ctx

    # This last step verifies that taking the max of the phase contexts is equivalent to taking the phase
    # dependant context for each phase
    for phase_offset, phase_ctx in enumerate(phase_ctxs):
        in_size = base_in_size + phase_offset
        assert (in_size - ctx) // params.stride_in == (in_size - phase_ctx) // params.stride_in

    return ctx
