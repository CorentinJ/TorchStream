import pytest

from tests.sliding_window_params_cases import (
    CONV_1D_PARAMS,
    EDGE_CASES_PARAMS,
    MOVING_AVERAGE_PARAMS,
    TRANSPOSED_CONV_1D_PARAMS,
)
from torchstream.sliding_window.nan_trick import get_context_size_empirically
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


@pytest.mark.parametrize(
    "sli_params,dilation",
    (CONV_1D_PARAMS[0] + TRANSPOSED_CONV_1D_PARAMS[0] + MOVING_AVERAGE_PARAMS[0] + EDGE_CASES_PARAMS[0]),
    ids=(CONV_1D_PARAMS[1] + TRANSPOSED_CONV_1D_PARAMS[1] + MOVING_AVERAGE_PARAMS[1] + EDGE_CASES_PARAMS[1]),
)
def test_context_matches_empirically(sli_params: SlidingWindowParams, dilation: int):
    # TODO! verify this with a transform instead of via get_nan_map
    ctx = get_context_size_empirically(sli_params)
    assert ctx == sli_params.streaming_context_size, (
        f"Failure for params {sli_params} -> real ctx={ctx}, params.ctx={sli_params.streaming_context_size}"
    )
