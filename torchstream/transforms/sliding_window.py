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


def overlap_windows(windows: np.ndarray, stride: int, overlap_fn=np.add) -> np.ndarray:
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
    overlap_fn.at(out, idx, windows)

    return out


def run_sliding_window(
    in_vec: np.ndarray,
    stride_in: int = 1,
    kernel_in: np.ndarray | None = None,
    kernel_in_fn=np.multiply,
    kernel_in_reduce=np.sum,
    stride_out: int = 1,
    kernel_out: np.ndarray | None = None,
    kernel_out_fn=np.multiply,
    overlap_fn=np.add,
):
    """
    Runs the given sliding window operation in 4 steps:
    - Applies the <kernel_in_fn> function to each window with the input kernel (defaults to element-wise
    multiplication with a kernel of ones)
    - Reduces each resulting window to a single scalar using the <kernel_in_reduce> function (defaults to summation)
    - Applies the <kernel_out_fn> function to each resulting scalar with the output kernel (defaults to element-wise
    multiplication with a kernel of ones)
    - Overlaps the resulting windows into a single output vector using the <overlap_fn> function (defaults to
    element-wise addition)
    """
    if kernel_in is None:
        kernel_in = np.ones((1,), dtype=in_vec.dtype)
    if kernel_out is None:
        kernel_out = np.ones((1,), dtype=in_vec.dtype)

    in_wins = kernel_in_fn(view_as_windows(in_vec, kernel_in.shape[0], stride_in), kernel_in)
    window_values = kernel_in_reduce(in_wins, axis=1, keepdims=True)
    out_wins = kernel_out_fn(window_values, kernel_out[None, :])
    out_vec = overlap_windows(out_wins, stride_out, overlap_fn=overlap_fn)

    return out_vec
