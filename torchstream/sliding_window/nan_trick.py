from typing import List, Tuple, Union

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


def get_nan_range(x: Union[torch.Tensor, np.ndarray], dim: int = -1):
    # TODO! doc
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    dim = range(x.ndim)[dim]
    x = x.mean(axis=tuple(i for i in range(x.ndim) if i != dim))

    corrupted_idx = np.where(np.isnan(x))[0]

    if not len(corrupted_idx):
        return None, None
    return corrupted_idx[0], corrupted_idx[-1] + 1


def check_nan_trick(
    sliding_window_params: SlidingWindowParams,
    in_len: int,
    out_len: int,
    nan_in_range: Tuple[int, int],
    nan_out_range: Tuple[int, int],
):
    # TODO! doc
    tsfm = DummySlidingWindowTransform(sliding_window_params)

    x = np.random.randn(in_len)
    set_nan_range(x, nan_in_range)

    out = tsfm(x)
    if len(out) != out_len:
        return False, f"expected out len {out_len}, got {len(out)}"

    left, right = get_nan_range(out)
    if (left, right) != nan_out_range:
        return False, f"expected out range {nan_out_range}, got {(left, right)}"

    return True, None


def _count_unique_np_arrays(arrays: List[np.ndarray]) -> List[Tuple[np.ndarray, int]]:
    # NOTE: this can also be achieved with np.unique given an axis arg, but it requires padding arrays to the same
    # shape...
    array_idx_map = {}
    array_count = []
    for array in arrays:
        key = array.tobytes()

        if key not in array_idx_map:
            array_idx_map[key] = len(array_count)
            array_count.append([array.copy(), 0])
        array_idx = array_idx_map[key]

        array_count[array_idx][1] += 1

    return list(map(tuple, array_count))


# TODO: name
def get_input_parameters_xxx(hypotheses: SlidingWindowParams):
    min_seq_size = max(sol.get_min_input_size() for sol in hypotheses)
    max_seq_size = min_seq_size + max(sol.kernel_size_in for sol in hypotheses)

    for seq_size in range(min_seq_size, max_seq_size + 1):
        outcomes = []
        for hypothesis in hypotheses:
            inv_map = hypothesis.get_inverse_map(seq_size)
            outcomes.append(inv_map)

        outcome_count = _count_unique_np_arrays(outcomes)
        3 + 2
