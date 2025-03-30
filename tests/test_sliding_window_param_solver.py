import pytest
import torch
from torch import nn

from tests.rng import set_seed
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params_for_transform
from torchstream.tensor_provider import TensorSpec


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("padding", [(0, 0), (2, 0), (0, 3), (1, 4)])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_conv1d(kernel_size: int, stride: int, padding: int, dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    if padding[0] >= kernel_span or padding[1] >= kernel_span:
        pytest.skip("Padding should be smaller than the kernel span")

    params_str = f"k={kernel_size}, s={stride}, p={padding}, d={dilation}"
    set_seed(0x5EED)

    conv = nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        # TODO: handle grouping?
    )

    def transform(x):
        # TODO: handle different padding modes
        x = torch.nn.functional.pad(x, padding)
        x = conv(x)
        return x

    sols = find_sliding_window_params_for_transform(transform, TensorSpec(shape=(1, 1, -1)))

    assert sols, f"Expected solution, but none found for {params_str}"
    assert any(
        sol.kernel_size_in == kernel_span
        and sol.stride_in == stride
        and sol.left_pad == padding[0]
        and sol.right_pad == padding[1]
        for sol in sols
    ), f"Expected solution with {params_str}, but none found in {sols}"
