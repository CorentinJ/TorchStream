from typing import Tuple

import numpy as np
from z3 import And, Not, Or, Solver, unsat

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


def get_init_kernel_array(kernel_size: int, full: bool = False) -> np.ndarray:
    """
    Initialize a default kernel sparsity array.

    A kernel array is an integer numpy array with the following convention:
    - 0: the kernel does not cover the element at that index
    - 1: unknown (may or may not cover the element)
    - 2: the kernel covers the element at that index

    By default, we set all elements to 1 (unknown) and force the first and last
    elements to 2 (covered) to ensure a minimum span. If full is True, all elements
    are set to 2 (covered).

    :param kernel_size: Span/size of the kernel.
    :return: A kernel sparsity array initialized with 1s, with the first and last elements set to 2.
    """
    kernel = np.ones(int(kernel_size), dtype=np.int64)
    if kernel.size > 0:
        kernel[0] = kernel[-1] = 2
    if full:
        kernel[:] = 2
    return kernel


def determine_kernel_sparsity(
    params: SlidingWindowParams,
    kernel_in_prior: np.ndarray,
    kernel_out_prior: np.ndarray,
    in_len: int,
    in_nan_range: Tuple[int, int],
    out_nan_idx: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    # TODO! doc
    if kernel_in_prior.shape != (params.kernel_size_in,):
        raise ValueError(f"kernel_in_prior must have shape ({params.kernel_size_in},), got {kernel_in_prior.shape}")
    if kernel_out_prior.shape != (params.kernel_size_out,):
        raise ValueError(f"kernel_out_prior must have shape ({params.kernel_size_out},), got {kernel_out_prior.shape}")

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
        any_corrupted_constraint = Or(
            *[
                And(corrupted_wins[win_idx], kernel_out[kernel_out_idx])
                for win_idx, in_start, _, kernel_out_idx in inv_map
            ]
        )
        solver.add(any_corrupted_constraint if out_idx in out_nan_idx else Not(any_corrupted_constraint))

    # If the solver can't find any solution, then the parameters do not allow to explain the observed nans
    if solver.check() == unsat:
        return None, None

    kernel_in_values = kernel_in_prior.copy()
    kernel_out_values = kernel_out_prior.copy()
    for i in range(params.kernel_size_in):
        if kernel_in_prior[i] == 1:
            if solver.check(kernel_in[i] == True) == unsat:
                kernel_in_values[i] = 0
            elif solver.check(kernel_in[i] == False) == unsat:
                kernel_in_values[i] = 2
    for i in range(params.kernel_size_out):
        if kernel_out_prior[i] == 1:
            if solver.check(kernel_out[i] == True) == unsat:
                kernel_out_values[i] = 0
            elif solver.check(kernel_out[i] == False) == unsat:
                kernel_out_values[i] = 2

    return kernel_in_values, kernel_out_values


### Bitvec version below - twice as slow :(   ####

# def np_to_bitvec(arr: np.ndarray, set_value: int) -> BitVec:
#     return BitVecVal(sum((1 << i) for i, bit in enumerate(arr) if bit == set_value), arr.size)


# def determine_kernel_sparsity(
#     params: SlidingWindowParams,
#     kernel_in_prior: np.ndarray,
#     kernel_out_prior: np.ndarray,
#     in_len: int,
#     in_nan_range: Tuple[int, int],
#     out_nan_idx: np.ndarray,
# ) -> Tuple[np.ndarray | None, np.ndarray | None]:
#     # TODO! doc
#     if kernel_in_prior.shape != (params.kernel_size_in,):
#         raise ValueError(f"kernel_in_prior must have shape ({params.kernel_size_in},), got {kernel_in_prior.shape}")
#     if kernel_out_prior.shape != (params.kernel_size_out,):
#         raise ValueError(f"kernel_out_prior must have shape ({params.kernel_size_out},), got {kernel_out_prior.shape}")

#     _, num_wins, _ = params.get_metrics_for_input(in_len)

#     solver = Solver()
#     corrupted_wins = BitVec("cw", num_wins)
#     kernel_in = BitVec("kiv", params.kernel_size_in)
#     kernel_out = BitVec("kov", params.kernel_size_out)

#     # Apply the kernel priors
#     ki_set_mask = np_to_bitvec(kernel_in_prior, 2)
#     ki_unset_mask = np_to_bitvec(kernel_in_prior, 0)
#     ko_set_mask = np_to_bitvec(kernel_out_prior, 2)
#     ko_unset_mask = np_to_bitvec(kernel_out_prior, 0)
#     solver.add(
#         (kernel_in & ki_set_mask) == ki_set_mask,
#         (kernel_in & ki_unset_mask) == 0,
#         (kernel_out & ko_set_mask) == ko_set_mask,
#         (kernel_out & ko_unset_mask) == 0,
#     )

#     for win_idx, ((in_start, in_stop), (out_start, out_stop)) in enumerate(params.iter_kernel_map(num_wins)):
#         win_mask = 1 << win_idx

#         # The kernel can only output nans (=be corrupted) if it has any overlap with the input nans
#         kernel_in_nan_range = (
#             int(max(in_nan_range[0], in_start) - in_start),
#             int(min(in_nan_range[1], in_stop) - in_start),
#         )
#         if in_nan_range[0] < in_stop and in_start < in_nan_range[1] and kernel_in_nan_range[0] < kernel_in_nan_range[1]:
#             assert 0 <= kernel_in_nan_range[0] < kernel_in_nan_range[1] <= params.kernel_size_in
#             kernel_slice = Extract(kernel_in_nan_range[1] - 1, kernel_in_nan_range[0], kernel_in)
#             # If any bit in the slice is set, then the window is corrupted
#             solver.add(UGT(kernel_slice, 0) == ((corrupted_wins & win_mask) != 0))
#         else:
#             solver.add(corrupted_wins & win_mask == 0)

#     for out_idx, inv_map in enumerate(params.get_inverse_kernel_map(in_len)):
#         any_corrupted_constraint = Or(
#             *[
#                 And(
#                     (corrupted_wins & (1 << win_idx)) != 0,
#                     (kernel_out & (1 << kernel_out_idx)) != 0,
#                 )
#                 for win_idx, in_start, _, kernel_out_idx in inv_map
#             ]
#         )
#         solver.add(any_corrupted_constraint if out_idx in out_nan_idx else Not(any_corrupted_constraint))

#     # If the solver can't find any solution, then the parameters do not allow to explain the observed nans
#     if solver.check() == unsat:
#         return None, None

#     kernel_in_values = kernel_in_prior.copy()
#     kernel_out_values = kernel_out_prior.copy()
#     for i in range(params.kernel_size_in):
#         if kernel_in_prior[i] == 1:
#             if solver.check(kernel_in & (1 << i) != 0) == unsat:
#                 kernel_in_values[i] = 0
#             elif solver.check(kernel_in & (1 << i) == 0) == unsat:
#                 kernel_in_values[i] = 2
#     for i in range(params.kernel_size_out):
#         if kernel_out_prior[i] == 1:
#             if solver.check(kernel_out & (1 << i) != 0) == unsat:
#                 kernel_out_values[i] = 0
#             elif solver.check(kernel_out & (1 << i) == 0) == unsat:
#                 kernel_out_values[i] = 2

#     return kernel_in_values, kernel_out_values


def get_nan_map(
    params: SlidingWindowParams,
    in_len: int,
    in_nan_range: Tuple[int, int] | None,
    kernel_in_prior: np.ndarray | None = None,
    kernel_out_prior: np.ndarray | None = None,
):
    # TODO! doc
    assert in_nan_range is None or (0 <= in_nan_range[0] < in_nan_range[1] <= in_len)

    if kernel_in_prior is None:
        kernel_in_prior = get_init_kernel_array(params.kernel_size_in, full=True)
    if kernel_out_prior is None:
        kernel_out_prior = get_init_kernel_array(params.kernel_size_out, full=True)

    if kernel_in_prior.shape != (params.kernel_size_in,):
        raise ValueError(f"kernel_in_prior must have shape ({params.kernel_size_in},), got {kernel_in_prior.shape}")
    if kernel_out_prior.shape != (params.kernel_size_out,):
        raise ValueError(f"kernel_out_prior must have shape ({params.kernel_size_out},), got {kernel_out_prior.shape}")

    _, num_wins, out_len = params.get_metrics_for_input(in_len)
    nan_map = np.zeros(out_len, dtype=np.int64)
    if not in_nan_range:
        return nan_map

    for (in_start, in_stop), (out_start, out_stop) in params.iter_kernel_map(num_wins):
        window_value = max(
            kernel_in_prior[i] if in_nan_range[0] <= in_start + i < in_nan_range[1] else 0
            for i in range(params.kernel_size_in)
        )
        for i in range(params.kernel_size_out):
            if 0 <= out_start + i < out_len:
                nan_map[out_start + i] = max(nan_map[out_start + i], min(kernel_out_prior[i], window_value))

    return nan_map
