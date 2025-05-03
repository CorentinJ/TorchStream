import logging
from typing import Callable, Optional, Tuple

import numpy as np

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.dummy_sliding_window_transform import DummySlidingWindowTransform
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

logger = logging.getLogger(__name__)


def get_nan_range(x: Sequence) -> Tuple[int, int] | None:
    # TODO! doc
    # TODO: numpy() function
    # TODO: is_nan -> any reduction instead
    x = x.data.mean(axis=tuple(i for i in range(x.ndim) if i != x.dim))

    corrupted_idx = np.where(np.isnan(x))[0]

    if not len(corrupted_idx):
        return None
    return corrupted_idx[0], corrupted_idx[-1] + 1


def run_nan_trick(
    trsfm: Callable,
    in_seq: Sequence,
    in_nan_range: Tuple[int, int],
    output_spec: Optional[SeqSpec] = None,
) -> Tuple[int, Tuple[int, int] | None]:
    """
    TODO: doc

    TODO: handle multi-input/output
    """
    if not in_seq.size:
        raise ValueError(f"Input sequence size must be greater than 0, got {in_seq.size}")
    if not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq.size):
        raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")
    output_spec = output_spec or in_seq.spec

    # Corrupt the given range of the input sequence with NaNs
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

        return 0, None

    out_nan_range = get_nan_range(out_seq)
    logger.debug(f"Got a {tuple(out_seq.shape)} shaped output with nans at {out_nan_range}")

    return out_seq.size, out_nan_range


def check_nan_trick(
    sliding_window_params: SlidingWindowParams,
    in_len: int,
    out_len: int,
    nan_in_range: Tuple[int, int],
    nan_out_range: Tuple[int, int] | None,
):
    # TODO! doc
    tsfm = DummySlidingWindowTransform(sliding_window_params)

    x = np.random.randn(in_len)
    x[nan_in_range] = float("nan")

    out = tsfm(x)
    if len(out) != out_len:
        return False, f"expected out len {out_len}, got {len(out)}"

    actual_nan_out_range = get_nan_range(out)
    if actual_nan_out_range != nan_out_range:
        return False, f"expected out range {nan_out_range}, got {actual_nan_out_range}"

    return True, None
