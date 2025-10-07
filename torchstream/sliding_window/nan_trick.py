import logging
from typing import Callable, Optional, Tuple

import numpy as np

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence

logger = logging.getLogger(__name__)


def get_seq_nan_idx(x: Sequence) -> np.ndarray:
    if not x.size:
        return np.empty(0, dtype=np.int64)

    # TODO! doc
    # TODO: numpy() function
    # TODO: is_nan -> any reduction instead
    x = x.data.mean(axis=tuple(i for i in range(x.ndim) if i != x.dim))

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

    out_nan_idx = get_seq_nan_idx(out_seq)
    logger.debug(f"Got a {tuple(out_seq.shape)} shaped output with nans at {out_nan_idx}")

    return out_seq, out_nan_idx
