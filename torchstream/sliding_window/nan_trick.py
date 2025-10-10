import logging
from typing import Callable, Optional, Tuple

import numpy as np

from torchstream.sequence.dtype import SeqArrayLike
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.kernel_sparsity import get_nan_map
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams, get_output_delay

logger = logging.getLogger(__name__)


def get_nan_idx(x: SeqArrayLike | Sequence, axis=None) -> np.ndarray:
    if isinstance(x, Sequence):
        axis = x.dim
        x = x.data
    elif axis is None:
        axis = -1

    if not x.shape[axis]:
        return np.empty(0, dtype=np.int64)

    # TODO! doc
    # TODO: numpy() function
    # TODO: is_nan -> any reduction instead
    x = x.mean(axis=tuple(i for i in range(x.ndim) if i != axis))

    return np.where(np.isnan(x))[0]


def run_nan_trick(
    trsfm: Callable,
    in_seq: Sequence,
    in_nan_range: Tuple[int, int] | None,
    out_spec: Optional[SeqSpec] = None,
    zero_size_exception_types: Tuple[type[Exception], ...] = (RuntimeError,),
) -> Tuple[Sequence, np.ndarray]:
    """
    TODO: doc

    TODO: handle multi-input/output
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
    out_seq = Sequence.apply(trsfm, in_seq, out_spec, zero_size_exception_types=zero_size_exception_types)

    out_nan_idx = get_nan_idx(out_seq)
    logger.debug(f"Got a {tuple(out_seq.shape)} shaped output with nans at {out_nan_idx}")

    return out_seq, out_nan_idx


def get_context_size_empirically(params: SlidingWindowParams):
    phase_ctxs = []
    for phase_offset in range(params.stride_in):
        # FIXME!
        in_size = 100 + phase_offset
        *_, out_size = params.get_metrics_for_input(in_size)
        out_delay = get_output_delay(params, in_size)
        stream_out_pos = out_size - out_delay
        assert stream_out_pos > 0, "Input size is not sufficient"

        for wins_to_keep in range(0, in_size):
            base_wins_to_drop = in_size // params.stride_in
            if stream_out_pos < (base_wins_to_drop - wins_to_keep) * params.stride_out:
                continue

            wins_to_drop = base_wins_to_drop - (wins_to_keep + 1)
            assert wins_to_drop > 0, "Input size is not sufficient"
            tsfm_out_pos = wins_to_drop * params.stride_out
            out_trim_start = stream_out_pos - tsfm_out_pos

            out_nan_map = get_nan_map(params, in_size, in_nan_range=(0, params.stride_in))
            assert len(out_nan_map[out_trim_start:]), "Input size is not sufficient"
            ctx_is_enough = out_nan_map[out_trim_start:].sum() == 0

            if ctx_is_enough:
                ctx = max(0, (wins_to_keep - 1) * params.stride_in + (in_size % params.stride_in) + 1)
                assert (in_size - ctx) // params.stride_in == base_wins_to_drop - wins_to_keep
                phase_ctxs.append(ctx)
                break

    ctx = max(phase_ctxs)

    for phase_offset, phase_ctx in enumerate(phase_ctxs):
        in_size = 100 + phase_offset
        assert (in_size - ctx) // params.stride_in == (in_size - phase_ctx) // params.stride_in

    assert min(phase_ctxs) + params.stride_in > ctx

    return ctx
