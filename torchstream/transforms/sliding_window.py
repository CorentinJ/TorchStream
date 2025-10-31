
import numpy as np


def view_as_windows(arr: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError("view_as_windows only supports 1D arrays")
    if (arr.size - kernel_size) % stride != 0:
        raise ValueError("Array size is not compatible with the given kernel size and stride")

    win_starts = np.arange(0, arr.size - kernel_size + 1, stride)
    offsets = np.arange(kernel_size)
    idx = win_starts[:, None] + offsets[None, :]
    return arr[idx]


def overlap_reduce(windows: np.ndarray, stride: int, reduction=np.add) -> np.ndarray:
    """
    Overlaps windows of the same size into a single array by applying a reduction operation. Common reductions
    are np.add, np.maximum, np.minimum.
    """
    if windows.ndim != 2:
        raise ValueError("overlap_reduce expects a 2D array of shape (num_windows, window_size)")

    n, k = windows.shape[:2]
    out_len = (n - 1) * stride + k
    out = np.zeros(out_len, dtype=windows.dtype)

    idx = np.arange(k) + np.arange(n)[:, None] * stride
    reduction.at(out, idx, windows)

    return out
