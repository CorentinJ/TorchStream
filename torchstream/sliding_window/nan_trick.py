from typing import Tuple, Union

import numpy as np
import torch

from torchstream.sliding_window.dummy_sliding_window_transform import DummySlidingWindowTransform
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


def set_nan_range(
    x: Union[torch.Tensor, np.ndarray],
    range: Union[slice, Tuple[int, int]],
    dim: int = -1,
) -> Union[torch.Tensor, np.ndarray]:
    # TODO! doc
    if not isinstance(range, slice):
        range = slice(*range)

    slices = [slice(None)] * x.ndim
    slices[dim] = range

    x[tuple(slices)] = float("nan")


def get_nan_range(x: Union[torch.Tensor, np.ndarray], dim: int = -1) -> Tuple[int, int] | None:
    # TODO! doc
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    dim = range(x.ndim)[dim]
    x = x.mean(axis=tuple(i for i in range(x.ndim) if i != dim))

    corrupted_idx = np.where(np.isnan(x))[0]

    if not len(corrupted_idx):
        return None
    return corrupted_idx[0], corrupted_idx[-1] + 1


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
    set_nan_range(x, nan_in_range)

    out = tsfm(x)
    if len(out) != out_len:
        return False, f"expected out len {out_len}, got {len(out)}"

    actual_nan_out_range = get_nan_range(out)
    if actual_nan_out_range != nan_out_range:
        return False, f"expected out range {nan_out_range}, got {actual_nan_out_range}"

    return True, None
