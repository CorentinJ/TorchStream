import logging
from typing import Callable, Optional, Tuple

import numpy as np

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

logger = logging.getLogger(__name__)


def get_out_nan_idx(x: Sequence) -> np.ndarray:
    # TODO! doc
    # TODO: numpy() function
    # TODO: is_nan -> any reduction instead
    x = x.data.mean(axis=tuple(i for i in range(x.ndim) if i != x.dim))

    return np.where(np.isnan(x))[0]


def run_nan_trick(
    trsfm: Callable,
    in_seq: Sequence,
    in_nan_range: Tuple[int, int] | None,
    output_spec: Optional[SeqSpec] = None,
) -> Tuple[int, np.ndarray]:
    """
    TODO: doc

    TODO: handle multi-input/output
    """
    if not in_seq.size:
        raise ValueError(f"Input sequence size must be greater than 0, got {in_seq.size}")
    if in_nan_range and not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq.size):
        raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")
    output_spec = output_spec or in_seq.spec

    # Corrupt the given range of the input sequence with NaNs
    if in_nan_range:
        in_seq[slice(*in_nan_range)] = float("nan")

    # Forward the input through the transform
    logger.debug(f"Running transform with input size {in_seq.size} and nans at {in_nan_range}")
    try:
        out_seq = Sequence.apply(trsfm, in_seq, output_spec)
    except RuntimeError as e:
        # We'll assume that RuntimeError are conv errors for a too small input size
        # TODO: more reliable mechanism
        # TODO: handle errors due to nans

        logger.info(f"Transformed failed with {repr(e)}")

        return 0, np.empty(0, dtype=np.int64)

    out_nan_idx = get_out_nan_idx(out_seq)
    logger.debug(f"Got a {tuple(out_seq.shape)} shaped output with nans at {out_nan_idx}")

    return out_seq.size, out_nan_idx


def check_nan_trick(
    params: SlidingWindowParams,
    in_len: int,
    out_len: int,
    nan_in_range: Tuple[int, int] | None,
    out_nan_idx: np.ndarray,
) -> bool:
    # TODO! doc
    nan_map = get_nan_map(params, in_len, nan_in_range)
    if out_len != len(nan_map):
        return False

    if (nan_map[out_nan_idx] == 0).any():
        return False

    nan_map[out_nan_idx] = 3
    if (nan_map == 2).any():
        return False

    return True


def get_nan_map(
    params: SlidingWindowParams,
    in_len: int,
    nan_in_range: Tuple[int, int] | None,
):
    # TODO! doc
    _, _, output_size = params.get_metrics_for_input(in_len)

    nan_map = np.zeros(output_size, dtype=np.int64)
    if not nan_in_range:
        return nan_map

    for in_sli, out_sli in params.get_kernel_map(in_len):
        if nan_in_range[0] < in_sli.stop and in_sli.start < nan_in_range[1]:
            out_sli = slice(
                max(0, out_sli.start),
                min(out_sli.stop, output_size),
            )
            nan_map[out_sli] = np.maximum(nan_map[out_sli], 1)
            if nan_in_range[0] <= in_sli.start < nan_in_range[1] or nan_in_range[0] < in_sli.stop <= nan_in_range[1]:
                nan_map[out_sli.start] = 2
                nan_map[out_sli.stop - 1] = 2

    return nan_map
