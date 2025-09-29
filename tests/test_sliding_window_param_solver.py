from typing import Tuple
from venv import logger

import numpy as np
import pytest
import torch
from torch import nn

from tests.rng import set_seed
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
    get_all_output_delays,
    get_canonicalized_in_out_size_params,
    get_streaming_context_size,
)
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params_for_transform
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent


def _get_sol_params(sol: SlidingWindowParams):
    return {
        "shape": get_canonicalized_in_out_size_params(sol),
        "min_in_size": sol.get_min_input_size(),
        "out_delays": get_all_output_delays(sol),
        "context_size": get_streaming_context_size(sol),
    }


def _find_solution_or_equivalent(transform, seq_spec, expected_sol):
    # TODO: let the solver print this
    logger.debug(
        f"Expected solution: {expected_sol}"
        f"\nwith shape {get_canonicalized_in_out_size_params(expected_sol)}"
        f"\nwith out delays {get_all_output_delays(expected_sol)}"
        f"\nwith context size {get_streaming_context_size(expected_sol)}"
    )

    sols = find_sliding_window_params_for_transform(
        transform, seq_spec, debug_ref_params=expected_sol, max_equivalent_sols=1
    )
    assert len(sols) == 1, f"Expected exactly one solution, got {len(sols)}: {sols}"

    if expected_sol not in sols:
        assert any(_get_sol_params(sol) == _get_sol_params(expected_sol) for sol in sols)
        logger.warning("Could not find the expected solution, but found an equivalent one")

    test_stream_equivalent(transform, SlidingWindowStream(transform, sols[0], seq_spec))


@pytest.mark.parametrize("padding", [(0, 0), (2, 0), (0, 3), (1, 4)])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 10, 17])
@pytest.mark.parametrize("stride", [1, 2, 3, 10, 17])
def test_conv_1d(kernel_size: int, stride: int, padding: Tuple[int, int], dilation: int):
    kernel_span = (kernel_size - 1) * dilation + 1
    if stride > kernel_span:
        pytest.skip("Stride should be smaller than the kernel span")
    if padding[0] >= kernel_span or padding[1] >= kernel_span:
        pytest.skip("Padding should be smaller than the kernel span")
    if kernel_size == 1 and dilation > 1:
        pytest.skip("Redundant")

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

    _find_solution_or_equivalent(transform, SeqSpec((1, 1, -1)), expected_sol)


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
            'Output trimming (named "padding" for transposed convs) should be smaller than the output kernel span'
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

    _find_solution_or_equivalent(conv, SeqSpec((1, 1, -1)), expected_sol)


# NOTE: our solver has the following modeling limitation: on any layer with an input stride > 1, the current combined
# stride of the model must be expressible as either 1 / x or x / 1. A couple of examples:
#   - L1: transposed conv with output stride = 3, L2: conv with input stride = 6
#       -> after L1 our combined stride is 3 (3 / 1 -> OK) and after L2 it's 3 / 6 = 1 / 2 -> OK. We can model this.
#   - L1: conv with input stride = 2, L2: conv with input stride = 3
#       -> after L1 our combined stride is 1 / 2, and after L2 it's 1 / 6. All OK
#   - L1: transposed conv with output stride = 3, L2: conv with input stride = 2
#       -> after L1 our combined stride is 3 (3 / 1 -> OK) and after L2 it's 3 / 2 -> NOT OK. The solver will fail.
#   - L1: conv with input stride = 2, L2: transposed conv with output stride = 3
#       -> After L1 our combined stride is 1 / 2, and after L2 it's 3 / 2 BUT the check only needs to hold on layers
#       with an input stride > 1. E.g. adding another conv with input stride = 2 as L3 will fail the solver.
# Note that in practice, models will almost always meet this requirement. Indeed, most models either only upsample,
# downsample, or downsample first before upsampling. Only in the case where a model upsamples before downsampling
# could we have this issue (provided the strides do not meet the condition) - and I don't know yet of such a model.
# TODO! This should be a comment within the solver
@pytest.mark.parametrize(
    "conv_params",
    [
        (
            [
                {"transposed": False, "kernel_size": 5, "stride": 2, "padding": 1},
                {"transposed": True, "kernel_size": 4, "stride": 3, "padding": 2},
            ],
            SlidingWindowParams(
                kernel_size_in=5, stride_in=2, left_pad=1, right_pad=1, kernel_size_out=4, stride_out=3, out_trim=2
            ),
        ),
        (
            [
                {"transposed": False, "kernel_size": 7, "stride": 1, "padding": 3},
                {"transposed": True, "kernel_size": 16, "stride": 8, "padding": 4},
            ],
            None,
        ),
    ],
)
def test_conv_mix(conv_params):
    set_seed(0x5EED)

    conv_params, expected_sol = conv_params

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
    _find_solution_or_equivalent(network, SeqSpec((1, 1, -1)), expected_sol)


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
