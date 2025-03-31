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
def test_conv_1d(kernel_size: int, stride: int, padding: int, dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    if padding[0] >= kernel_span or padding[1] >= kernel_span:
        pytest.skip("Padding should be smaller than the kernel span")

    params_str = f"k={kernel_span} (kernel {kernel_size} with d={dilation}), s={stride}, p={padding}"
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


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("padding", [0])  # , 1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_conv_transpose_1d(kernel_size: int, stride: int, padding: int, dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    # See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    # effective_padding = dilation * (kernel_size - 1) - padding
    # if effective_padding < 0:
    # pytest.skip("Effective padding should be positive")
    effective_padding = 0

    params_str = (
        f"k={kernel_span} (kernel {kernel_size} with d={dilation}), s={stride}, p={effective_padding} (arg={padding})"
    )
    set_seed(0x5EED)

    conv = nn.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        # TODO: handle grouping?
        # TODO: handle output padding?
    )

    sols = find_sliding_window_params_for_transform(conv, TensorSpec(shape=(1, 1, -1)))

    assert sols, f"Expected solution, but none found for {params_str}"
    assert any(
        sol.kernel_size_out == kernel_span
        and sol.stride_out == stride
        and sol.left_pad == sol.right_pad == effective_padding
        for sol in sols
    ), f"Expected solution with {params_str}, but none found in {sols}"
