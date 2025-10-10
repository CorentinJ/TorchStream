import logging

import pytest
import torch
from torch import nn

from tests.rng import set_seed
from tests.sliding_window_params_cases import (
    CONV_1D_PARAMS,
    EDGE_CASES_PARAMS,
    MOVING_AVERAGE_PARAMS,
    TRANSPOSED_CONV_1D_PARAMS,
)
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.dummy_sliding_window_transform import DummySlidingWindowTransform
from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
)
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("sli_params,dilation", CONV_1D_PARAMS[0], ids=CONV_1D_PARAMS[1])
def test_conv_1d(sli_params: SlidingWindowParams, dilation: int):
    set_seed(0x5EED)

    conv1d_ki = (sli_params.kernel_size_in - 1) // dilation + 1
    conv = nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=conv1d_ki,
        stride=sli_params.stride_in,
        dilation=dilation,
        # TODO: handle grouping?
    )

    def transform(x):
        # TODO: handle different padding modes
        x = torch.nn.functional.pad(x, (sli_params.left_pad, sli_params.right_pad))
        x = conv(x)
        return x

    conv_stream = SlidingWindowStream(
        transform,
        sli_params,
        SeqSpec(1, 1, -1),
    )

    test_stream_equivalent(
        transform,
        conv_stream,
        throughput_check_max_delay=0,
    )


@pytest.mark.parametrize("sli_params,dilation", TRANSPOSED_CONV_1D_PARAMS[0], ids=TRANSPOSED_CONV_1D_PARAMS[1])
def test_conv_transpose_1d(sli_params: SlidingWindowParams, dilation: int):
    set_seed(0x5EED)

    tconv1d_ko = (sli_params.kernel_size_out - 1) // dilation + 1
    conv = nn.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=tconv1d_ko,
        stride=sli_params.stride_out,
        # "padding" is poorly explained in https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
        # A better explanation of the parameter is that it trims the output on both sides by the given amount.
        padding=sli_params.out_trim,
        dilation=dilation,
        # TODO: handle grouping?
        # TODO: handle output padding?
    )

    conv_stream = SlidingWindowStream(
        conv,
        sli_params,
        SeqSpec(1, 1, -1),
    )

    test_stream_equivalent(
        conv,
        conv_stream,
        throughput_check_max_delay=sli_params.out_trim,
    )


@pytest.mark.parametrize("sli_params,dilation", MOVING_AVERAGE_PARAMS[0], ids=MOVING_AVERAGE_PARAMS[1])
def test_moving_average(sli_params: SlidingWindowParams, dilation: int):
    tsfm = DummySlidingWindowTransform(sli_params)

    tsfm_stream = SlidingWindowStream(
        tsfm,
        sli_params,
        SeqSpec(-1, dtype=float),
    )

    test_stream_equivalent(
        tsfm,
        tsfm_stream,
        throughput_check_max_delay=sli_params.out_trim,
    )


@pytest.mark.parametrize("sli_params,dilation", EDGE_CASES_PARAMS[0], ids=EDGE_CASES_PARAMS[1])
def test_edge_cases(sli_params: SlidingWindowParams, dilation: int):
    tsfm = DummySlidingWindowTransform(sli_params)

    tsfm_stream = SlidingWindowStream(
        tsfm,
        sli_params,
        SeqSpec(-1, dtype=float),
    )

    test_stream_equivalent(
        tsfm,
        tsfm_stream,
        throughput_check_max_delay=sli_params.out_trim,
    )
