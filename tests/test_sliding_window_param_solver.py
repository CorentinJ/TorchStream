import logging
from typing import List, Tuple

import numpy as np
import pytest
import torch
from torch import nn

from tests.rng import set_seed
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params_for_transform


# FIXME!!
@pytest.mark.parametrize("kernel_size", [1, 2, 3])  # , 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3])  # , 10, 17])
@pytest.mark.parametrize("padding", [(0, 0), (2, 0)])  # , (0, 3), (1, 4)])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_conv_1d(kernel_size: int, stride: int, padding: Tuple[int, int], dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    if padding[0] >= kernel_span or padding[1] >= kernel_span:
        pytest.skip("Padding should be smaller than the kernel span")

    set_seed(0x5EED)

    expected_sol = SlidingWindowParams(
        kernel_size_in=kernel_span,
        stride_in=stride,
        left_pad=padding[0],
        right_pad=padding[1],
    )

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

    sols = find_sliding_window_params_for_transform(transform, SeqSpec((1, 1, -1)))
    assert expected_sol in sols


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("padding", [1, 2, 5])  # FIXME!!
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_conv_transpose_1d(kernel_size: int, stride: int, padding: int, dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    if padding >= kernel_span:
        pytest.skip(
            "Output trimming (named padding for transposed convs) should be smaller than the output kernel span"
        )

    set_seed(0x5EED)

    expected_sol = SlidingWindowParams(
        kernel_size_out=kernel_span,
        stride_out=stride,
        out_trim=padding,
    )

    # The torch docs poorly explain the mechanism of transposed convolutions. Here's my take:
    # Each individual input element multiplies the kernel (element-wise). That output is offset by the stride on each
    # step, and all resulting vectors are summed.
    conv = nn.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=stride,
        # "padding" is poorly explained in https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
        # A better explanation of the parameter is that it trims the output on both sides by the given amount.
        padding=padding,
        dilation=dilation,
        # TODO: handle grouping?
        # TODO: handle output padding?
    )

    sols = find_sliding_window_params_for_transform(conv, SeqSpec((1, 1, -1)))
    assert expected_sol in sols


@pytest.mark.parametrize(
    "conv_params",
    [
        # (
        #     {"transposed": False, "kernel_size": 7, "stride": 1, "padding": 3, "dilation": 1},
        #     {"transposed": False, "kernel_size": 3, "stride": 1, "padding": 2, "dilation": 2},
        #     {"transposed": True, "kernel_size": 2, "stride": 2},
        #     {"transposed": True, "kernel_size": 3, "stride": 3},
        # ),
        (
            {"transposed": False, "kernel_size": 7, "stride": 1, "padding": 3},
            {"transposed": True, "kernel_size": 16, "stride": 8, "padding": 4},
        ),
    ],
)
def test_conv_mix(conv_params: List[Tuple[dict]]):
    set_seed(0x5EED)

    network = nn.Sequential(
        *[
            (nn.ConvTranspose1d if params["transposed"] else nn.Conv1d)(
                in_channels=1,
                out_channels=1,
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params.get("padding", 0),
                dilation=params.get("dilation", 1),
            )
            for params in conv_params
        ]
    )
    sols = find_sliding_window_params_for_transform(network, SeqSpec((1, 1, -1)))
    logging.info(sols)

    # TODO: include expected parameters
    assert sols, f"Expected solution, but none found for {network}"


def test_infinite_receptive_field():
    """
    Tests that the solver does not find a solution for a transform that has an infinite receptive field.
    """

    def transform(x: np.ndarray):
        return np.full_like(x, fill_value=np.mean(x))

    sols = find_sliding_window_params_for_transform(transform, SeqSpec((-1,), dtype=np.float32))
    assert not sols


def test_no_receptive_field():
    """
    Tests that the solver does not find a solution for a transform that has no receptive field (output is not
    a function of the input).
    """

    def transform(x: np.ndarray):
        return np.full_like(x, fill_value=3.0)

    sols = find_sliding_window_params_for_transform(transform, SeqSpec((-1,), dtype=np.float32))
    assert not sols


@pytest.mark.parametrize("variant", ["prefix_mean", "suffix_mean", "mod7"])
def test_variable_receptive_field(variant: str):
    """
    Tests that the solver does not find a solution for a transform that has a receptive field of variable size.

    NOTE: these could technically be handled
    """
    if variant in ("prefix_mean", "suffix_mean"):
        pytest.skip("These cases are not properly handled yet")

    def transform(x: np.ndarray):
        y = np.zeros_like(x)

        if variant == "prefix_mean":
            for i in range(len(x)):
                y[i] = np.mean(x[: i + 1])
        elif variant == "suffix_mean":
            for i in range(len(x)):
                y[i] = np.mean(x[i:])
        elif variant == "mod7":
            for i in range(len(x)):
                ksize = (i % 7) + 1
                y[i] = np.mean(x[i : i + ksize])

        return y

    sols = find_sliding_window_params_for_transform(transform, SeqSpec((-1,), dtype=np.float32))
    assert not sols
