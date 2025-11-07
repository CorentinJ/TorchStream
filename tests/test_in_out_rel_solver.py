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
from torchstream.sliding_window.sliding_window_params_solver import (
    SlidingWindowParamsSolver,
)


def _find_in_out_size(transform, seq_spec, expected_sol):
    solver = SlidingWindowParamsSolver(transform, seq_spec, debug_ref_params=expected_sol)
    in_out_size_params = solver.find_in_out_size_params()
    if expected_sol:
        assert in_out_size_params == expected_sol.canonicalized_in_out_shape_params


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

    _find_in_out_size(transform, SeqSpec(1, 1, -1), sli_params)


@pytest.mark.parametrize("sli_params,dilation", TRANSPOSED_CONV_1D_PARAMS[0], ids=TRANSPOSED_CONV_1D_PARAMS[1])
def test_conv_transpose_1d(sli_params: SlidingWindowParams, dilation: int):
    set_seed(0x5EED)

    # The torch docs poorly explain the mechanism of transposed convolutions. Here's my take:
    # Each individual input element multiplies the kernel (element-wise). That output is offset by the stride on each
    # step, and all resulting vectors are summed.
    tconv1d_ko = (sli_params.kernel_size_out - 1) // dilation + 1
    conv = nn.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=tconv1d_ko,
        stride=sli_params.stride_out,
        # "padding" is poorly explained in https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
        # A better explanation of the parameter is that it trims the output on both sides by the given amount.
        padding=sli_params.left_out_trim,
        # "output_padding" increases the output size on the right by the given amount, hence it's the inverse of 
        # right_out_trim. However we already trim on both sides by "padding=sli_params.left_out_trim" (see above), 
        # so we need to counterbalance by re-adding left_out_trim.
        output_padding=sli_params.left_out_trim - sli_params.right_out_trim,
        dilation=dilation,
        # TODO: handle grouping?
    )

    _find_in_out_size(conv, SeqSpec(1, 1, -1), sli_params)


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
                kernel_size_in=5,
                stride_in=2,
                left_pad=1,
                right_pad=1,
                kernel_size_out=4,
                stride_out=3,
                left_out_trim=2,
                right_out_trim=2,
            ),
        ),
        (
            [
                {"transposed": False, "kernel_size": 5, "stride": 2, "padding": 1},
                {"transposed": True, "kernel_size": 10, "stride": 3, "padding": 2},
            ],
            SlidingWindowParams(
                kernel_size_in=5,
                stride_in=2,
                left_pad=1,
                right_pad=1,
                kernel_size_out=10,
                stride_out=3,
                left_out_trim=2,
                right_out_trim=2,
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
    _find_in_out_size(network, SeqSpec(1, 1, -1), expected_sol)


@pytest.mark.parametrize("sli_params,dilation", MOVING_AVERAGE_PARAMS[0], ids=MOVING_AVERAGE_PARAMS[1])
def test_moving_average(sli_params: SlidingWindowParams, dilation: int):
    tsfm = DummySlidingWindowTransform(sli_params)

    _find_in_out_size(tsfm, SeqSpec(-1, dtype=float), sli_params)


@pytest.mark.parametrize("sli_params,dilation", EDGE_CASES_PARAMS[0], ids=EDGE_CASES_PARAMS[1])
def test_edge_cases(sli_params: SlidingWindowParams, dilation: int):
    set_seed(0x5EED)

    transform = DummySlidingWindowTransform(sli_params)
    _find_in_out_size(transform, SeqSpec(-1, dtype=float), sli_params)
