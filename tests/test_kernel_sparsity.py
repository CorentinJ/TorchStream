import numpy as np
import pytest

from tests.rng import set_seed
from tests.sliding_window_params_cases import MOVING_AVERAGE_PARAMS
from torchstream.sliding_window.kernel_sparsity import (
    KernelSparsitySampler,
    get_nan_map,
)
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


def _seed_from_params(params) -> int:
    return (
        params.kernel_size_in
        + 3 * params.stride_in
        + 5 * params.left_pad
        + 7 * params.right_pad
        + 11 * params.kernel_size_out
        + 13 * params.stride_out
        + 17 * params.out_trim
    )


def _generate_random_kernel(rng: np.random.Generator, size: int) -> np.ndarray:
    kernel = np.full(size, 2, dtype=np.int64)
    if size > 2:
        kernel[1:-1] = rng.choice([0, 2], size=size - 2)
    return kernel


@pytest.mark.parametrize("sli_params,_", MOVING_AVERAGE_PARAMS[0], ids=MOVING_AVERAGE_PARAMS[1])
def test_determine_kernel_sparsity_moving_average(sli_params: SlidingWindowParams, _):
    set_seed(0x5EED)
    rng = np.random.default_rng(_seed_from_params(sli_params))

    kernel_in_truth = _generate_random_kernel(rng, sli_params.kernel_size_in)
    kernel_out_truth = _generate_random_kernel(rng, sli_params.kernel_size_out)

    kernel_sparsity_sampler = KernelSparsitySampler(sli_params)
    for _ in range(10):
        in_len = sli_params.get_min_input_size_for_out_size(rng.integers(1, 100))
        in_nan_range = tuple(sorted(rng.choice(in_len + 1, size=2, replace=False)))
        in_nan_range = (int(in_nan_range[0]), int(in_nan_range[1]))

        out_nan_map = get_nan_map(
            sli_params,
            in_len,
            in_nan_range,
            kernel_in=kernel_in_truth,
            kernel_out=kernel_out_truth,
        )
        out_nan_idx = np.flatnonzero(out_nan_map)

        kernel_sparsity_sampler.add_in_out_map(in_len, in_nan_range, out_nan_idx)
        assert kernel_sparsity_sampler.has_solution(), "Failed to solve a valid case"

    kernel_in_posterior, kernel_out_posterior = kernel_sparsity_sampler.determine()

    in_compatible = np.logical_or(kernel_in_posterior == 1, kernel_in_posterior == kernel_in_truth)
    out_compatible = np.logical_or(kernel_out_posterior == 1, kernel_out_posterior == kernel_out_truth)
    assert np.all(in_compatible), "Posterior input kernel is incompatible with the ground truth"
    assert np.all(out_compatible), "Posterior output kernel is incompatible with the ground truth"

    assert np.sum(kernel_in_posterior == 1) <= max(0, sli_params.kernel_size_in - 2), (
        "No elements of the input kernel discovered"
    )
    assert np.sum(kernel_out_posterior == 1) <= max(0, sli_params.kernel_size_out - 2), (
        "No elements of the output kernel discovered"
    )
