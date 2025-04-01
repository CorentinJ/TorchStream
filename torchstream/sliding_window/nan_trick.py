import logging
from typing import Callable, Tuple, Union

import numpy as np
import torch

from torchstream.sliding_window.dummy_sliding_window_transform import DummySlidingWindowTransform
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.tensor_provider import TensorProvider

logger = logging.getLogger(__name__)


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


def run_nan_trick(
    trsfm: Callable,
    # TODO: more convenient signature
    input_provider: TensorProvider,
    in_seq_size: int,
    in_nan_range: Tuple[int, int],
) -> Tuple[int, Tuple[int, int] | None]:
    if in_seq_size < 1:
        raise ValueError(f"Input sequence size must be greater than 0, got {in_seq_size}")
    if not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq_size):
        raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")

    x = input_provider.get_tensor(in_seq_size)
    # TODO: move to TensorProvider
    assert x.shape[input_provider.dim] == in_seq_size

    set_nan_range(x, in_nan_range, dim=input_provider.dim)

    logger.debug(f"Running transform with input size {in_seq_size} and nans at {in_nan_range}")
    try:
        # FIXME: output format
        y = trsfm(x)
    except RuntimeError as e:
        # We'll assume that RuntimeError are conv errors for a too small input size
        # TODO: more reliable mechanism
        # TODO: handle errors due to nans

        logger.info(f"Transformed failed with {repr(e)}")

        return 0, None

    # FIXME: dim
    out_size = y.shape[-1]
    out_nan_range = get_nan_range(y, dim=-1)
    logger.debug(f"Got a {tuple(y.shape)} shaped output with nans at {out_nan_range}")

    return out_size, out_nan_range


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
