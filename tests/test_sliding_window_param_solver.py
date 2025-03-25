import pytest
import torch
from torch import nn

from tests.rng import set_seed
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params_for_transform
from torchstream.tensor_provider import TensorSpec

temp = nn.Conv1d(1, 1, kernel_size=7, stride=2, padding=0)


# TODO
def my_transform(x):
    x = torch.nn.functional.pad(x, (1, 3))
    return temp(x)


# kernel_size = 17, stride = 1, padding = 0, dilation = 1
# kernel_size = 5, stride = 1, padding = 0, dilation = 1
# 1 0 3 3
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 4, 5, 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5, 10, 17])
@pytest.mark.parametrize("padding", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_conv1d(kernel_size: int, stride: int, padding: int, dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    if padding >= kernel_span:
        pytest.skip("Padding should be smaller than the kernel span")

    params_str = f"k={kernel_size}, s={stride}, p={padding}, d={dilation}"
    set_seed(0x5EED)

    conv = nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        # TODO: handle grouping?
        # TODO: handle different padding modes
    )

    sols = find_sliding_window_params_for_transform(conv, TensorSpec(shape=(1, 1, -1)))

    assert sols, f"Expected solution, but none found for {params_str}"
    best_sol = sols[0]
    assert best_sol.kernel_size_in == kernel_span, f"Expected kernel span {kernel_span}, got {best_sol.kernel_size_in}"
    assert best_sol.stride_in == stride, f"Expected stride {stride}, got {best_sol.stride_in}"
    assert best_sol.left_pad == best_sol.right_pad == padding, (
        f"Expected padding {(padding, padding)}, got ({best_sol.left_pad}, {best_sol.right_pad})"
    )
