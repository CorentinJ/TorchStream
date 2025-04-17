from typing import Tuple

import pytest
import torch
from torch import nn

from tests.rng import set_seed
from tests.stream_equivalency import test_stream_equivalent
from torchstream.sequence_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("padding", [(0, 0), (1, 1), (2, 0), (0, 3), (1, 4)])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_conv_1d(kernel_size: int, stride: int, padding: Tuple[int, int], dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    if padding[0] >= kernel_span or padding[1] >= kernel_span:
        pytest.skip("Padding should be smaller than the kernel span")

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

    sliding_window_params = SlidingWindowParams(
        kernel_size_in=kernel_span,
        stride_in=stride,
        left_pad=padding[0],
        right_pad=padding[1],
    )

    conv_stream = SlidingWindowStream(
        transform,
        sliding_window_params,
        SeqSpec((1, 1, -1)),
    )

    test_stream_equivalent(conv_stream, transform)


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_conv_transpose_1d(kernel_size: int, stride: int, dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")

    set_seed(0x5EED)

    conv = nn.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        # TODO: (input) padding as explained in https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
        # seems completely wrong. Note that transposed convolutions have an input kernel size of 1, so it makes no
        # sense to have any input padding at all.
        # TODO: handle grouping?
        # TODO: handle output padding?
    )

    sliding_window_params = SlidingWindowParams(kernel_size_out=kernel_span, stride_out=stride)

    conv_stream = SlidingWindowStream(
        conv,
        sliding_window_params,
        SeqSpec((1, 1, -1)),
    )

    test_stream_equivalent(conv_stream, conv)
