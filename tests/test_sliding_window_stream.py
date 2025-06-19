from typing import Tuple

import numpy as np
import pytest
import torch
from torch import nn

from tests.rng import set_seed
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.dummy_sliding_window_transform import DummySlidingWindowTransform
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent


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
    if kernel_size == 1 and dilation > 1:
        pytest.skip("Redundant")

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

    test_stream_equivalent(
        transform,
        conv_stream,
        throughput_check_max_delay=0,
    )


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("out_trim", [0, 1, 2, 8, 9])
def test_conv_transpose_1d(kernel_size: int, stride: int, dilation: int, out_trim: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    if out_trim >= kernel_span:
        pytest.skip("Output trim should be smaller than the kernel span")
    if kernel_size == 1 and dilation > 1:
        pytest.skip("Redundant")

    set_seed(0x5EED)

    conv = nn.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=stride,
        # "padding" is poorly explained in https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
        # A better explanation of the parameter is that it trims the output on both sides by the given amount.
        padding=out_trim,
        dilation=dilation,
        # TODO: handle grouping?
        # TODO: handle output padding?
    )

    sliding_window_params = SlidingWindowParams(kernel_size_out=kernel_span, stride_out=stride, out_trim=out_trim)

    conv_stream = SlidingWindowStream(
        conv,
        sliding_window_params,
        SeqSpec((1, 1, -1)),
    )

    test_stream_equivalent(
        conv,
        conv_stream,
        throughput_check_max_delay=out_trim,
    )


@pytest.mark.parametrize("kernel_size_in", [1, 2, 5, 10])
@pytest.mark.parametrize("stride_in", [1, 2, 3, 10])
@pytest.mark.parametrize("padding", [(0, 0), (2, 0), (0, 3), (1, 4)])
@pytest.mark.parametrize("kernel_size_out", [1, 2, 4, 7])
@pytest.mark.parametrize("stride_out", [1, 2, 6, 7])
def test_moving_average(
    kernel_size_in: int,
    stride_in: int,
    padding: Tuple[int, int],
    kernel_size_out: int,
    stride_out: int,
):
    if stride_in > kernel_size_in or stride_out > kernel_size_out:
        pytest.skip("Stride should be smaller than the kernel span")
    if padding[0] >= kernel_size_in or padding[1] >= kernel_size_in:
        pytest.skip("Padding should be smaller than the kernel span")

    sliding_window_params = SlidingWindowParams(
        kernel_size_in=kernel_size_in,
        stride_in=stride_in,
        left_pad=padding[0],
        right_pad=padding[1],
        kernel_size_out=kernel_size_out,
        stride_out=stride_out,
    )
    tsfm = DummySlidingWindowTransform(sliding_window_params)

    tsfm_stream = SlidingWindowStream(
        tsfm,
        sliding_window_params,
        SeqSpec((-1,), dtype=np.float32),
    )

    test_stream_equivalent(
        tsfm,
        tsfm_stream,
        throughput_check_max_delay=0,
    )
