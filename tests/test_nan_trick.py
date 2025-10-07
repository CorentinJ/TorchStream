from typing import Tuple

import numpy as np
import pytest
import torch

from torchstream.sequence.dtype import SeqDTypeLike
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.nan_trick import get_nan_idx


@pytest.mark.parametrize(
    "shape, nan_range",
    [
        # TODO: add more cases
        ((1, 1, -1), None),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
    ],
)
def test_get_nan_range(
    shape: Tuple[int],
    dtype: SeqDTypeLike,
    nan_range: Tuple[int, int] | None,
):
    seq = Sequence.zeros(SeqSpec(shape, dtype=dtype), seq_size=20)

    if nan_range is not None:
        seq[nan_range] = float("nan")
    # FIXME!!
    assert get_nan_idx(seq) == nan_range
