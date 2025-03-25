from typing import Tuple

import numpy as np
import pytest
import torch

from torchstream.sliding_window.nan_trick import get_nan_range, set_nan_range


@pytest.mark.parametrize(
    "shape, dim, nan_range",
    [
        ((1, 1, 1), -1, None),
    ],
)
def test_get_nan_range(
    shape: Tuple[int],
    dim: int,
    nan_range: Tuple[int, int] | None,
):
    for tensor in [torch.randn(shape), np.random.randn(*shape)]:
        if nan_range is not None:
            set_nan_range(tensor, range=nan_range, dim=dim)
        assert get_nan_range(tensor, dim=dim) == nan_range
