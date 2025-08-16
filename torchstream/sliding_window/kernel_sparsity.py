from typing import Tuple

import numpy as np
from z3 import And, Bool, Not, Or, Solver, unsat

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


def get_init_kernel_array(kernel_size: int) -> np.ndarray:
    """
    Initialize a default kernel sparsity array.

    A kernel array is an integer numpy array with the following convention:
    - 0: the kernel does not cover the element at that index
    - 1: unknown (may or may not cover the element)
    - 2: the kernel covers the element at that index

    By default, we set all elements to 1 (unknown) and force the first and last
    elements to 2 (covered) to ensure a minimum span.

    :param kernel_size: Span/size of the kernel.
    :return: A kernel sparsity array initialized with 1s, with the first and last elements set to 2.
    """
    kernel = np.ones(int(kernel_size), dtype=np.int64)
    if kernel.size > 0:
        kernel[0] = kernel[-1] = 2
    return kernel


# TODO: tests
def determine_kernel_sparsity(
    params: SlidingWindowParams,
    kernel_in_prior: np.ndarray,
    kernel_out_prior: np.ndarray,
    in_len: int,
    in_nan_range: Tuple[int, int],
    out_nan_idx: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    # TODO! doc

    _, num_wins, _ = params.get_metrics_for_input(in_len)

    solver = Solver()
    corrupted_wins = [Bool("corrupted_win_" + str(i)) for i in range(num_wins)]
    kernel_in = [Bool("kernel_in_" + str(i)) for i in range(params.kernel_size_in)]
    kernel_out = [Bool("kernel_out_" + str(i)) for i in range(params.kernel_size_out)]

    # Apply the kernel priors
    for idx, val in enumerate(kernel_in_prior):
        if val == 0:
            solver.add(kernel_in[idx] == False)
        elif val == 2:
            solver.add(kernel_in[idx] == True)
    for idx, val in enumerate(kernel_out_prior):
        if val == 0:
            solver.add(kernel_out[idx] == False)
        elif val == 2:
            solver.add(kernel_out[idx] == True)

    for win_idx, ((in_start, in_stop), (out_start, out_stop)) in enumerate(params.iter_kernel_map(num_wins)):
        # The kernel can only output nans (=be corrupted) if it has any overlap with the input nans
        if in_nan_range[0] < in_stop and in_start < in_nan_range[1]:
            kernel_in_nan_range = (
                max(in_nan_range[0], in_start) - in_start,
                min(in_nan_range[1], in_stop) - in_start,
            )
            corrupted_wins[win_idx] = Or(*[kernel_in[i] for i in range(*kernel_in_nan_range)])
        else:
            solver.add(corrupted_wins[win_idx] == False)

    for out_idx, inv_map in enumerate(params.get_inverse_kernel_map(in_len)):
        if out_idx in out_nan_idx:
            solver.add(
                Or(
                    *[
                        And(corrupted_wins[win_idx], kernel_out[kernel_out_idx])
                        for win_idx, in_start, _, kernel_out_idx in inv_map
                    ]
                )
            )
        else:
            solver.add(
                And(
                    *[
                        Not(And(corrupted_wins[win_idx], kernel_out[kernel_out_idx]))
                        for win_idx, in_start, _, kernel_out_idx in inv_map
                    ]
                )
            )

    if solver.check() == unsat:
        return None, None

    model = solver.model()
    kernel_in_values = np.zeros(params.kernel_size_in, dtype=np.int64)
    kernel_out_values = np.zeros(params.kernel_size_out, dtype=np.int64)
    for i in range(params.kernel_size_in):
        if model[kernel_in[i]] == True:
            kernel_in_values[i] = 2
        elif model[kernel_in[i]] == False:
            kernel_in_values[i] = 0
        else:
            kernel_in_values[i] = 1
    for i in range(params.kernel_size_out):
        if model[kernel_out[i]] == True:
            kernel_out_values[i] = 2
        elif model[kernel_out[i]] == False:
            kernel_out_values[i] = 0
        else:
            kernel_out_values[i] = 1

    return kernel_in_values, kernel_out_values
