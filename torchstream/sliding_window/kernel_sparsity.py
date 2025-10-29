import logging
from typing import Tuple

import numpy as np
from z3 import UGT, And, BitVec, BitVecVal, Extract, Not, Or, Solver, unsat

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

logger = logging.getLogger(__name__)


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


def np_to_bitvec(arr: np.ndarray, set_value: int) -> BitVec:
    return BitVecVal(sum((1 << i) for i, bit in enumerate(arr) if bit == set_value), arr.size)


def get_nan_map(
    params: SlidingWindowParams,
    in_len: int,
    in_nan_range: Tuple[int, int] | None,
    kernel_in: np.ndarray | None = None,
    kernel_out: np.ndarray | None = None,
):
    # TODO! doc
    assert in_nan_range is None or (0 <= in_nan_range[0] < in_nan_range[1] <= in_len)

    if kernel_in is None:
        kernel_in = get_init_kernel_array(params.kernel_size_in, full=True)
    if kernel_out is None:
        kernel_out = get_init_kernel_array(params.kernel_size_out, full=True)

    if kernel_in.shape != (params.kernel_size_in,):
        raise ValueError(f"kernel_in_prior must have shape ({params.kernel_size_in},), got {kernel_in.shape}")
    if kernel_out.shape != (params.kernel_size_out,):
        raise ValueError(f"kernel_out_prior must have shape ({params.kernel_size_out},), got {kernel_out.shape}")

    _, num_wins, out_len = params.get_metrics_for_input(in_len)
    nan_map = np.zeros(out_len, dtype=np.int64)
    if not in_nan_range:
        return nan_map

    for (in_start, in_stop), (out_start, out_stop) in params.iter_kernel_map(num_wins):
        window_value = max(
            kernel_in[i] if in_nan_range[0] <= in_start + i < in_nan_range[1] else 0
            for i in range(params.kernel_size_in)
        )
        for i in range(params.kernel_size_out):
            if 0 <= out_start + i < out_len:
                nan_map[out_start + i] = max(nan_map[out_start + i], min(kernel_out[i], window_value))

    return nan_map


class KernelSparsitySampler:
    def __init__(
        self,
        params: SlidingWindowParams,
        kernel_in_prior: np.ndarray | None = None,
        kernel_out_prior: np.ndarray | None = None,
    ):
        # TODO! doc
        if kernel_in_prior is None:
            kernel_in_prior = get_init_kernel_array(params.kernel_size_in)
        if kernel_out_prior is None:
            kernel_out_prior = get_init_kernel_array(params.kernel_size_out)

        if kernel_in_prior.shape != (params.kernel_size_in,):
            raise ValueError(f"kernel_in_prior must have shape ({params.kernel_size_in},), got {kernel_in_prior.shape}")
        if kernel_out_prior.shape != (params.kernel_size_out,):
            raise ValueError(
                f"kernel_out_prior must have shape ({params.kernel_size_out},), got {kernel_out_prior.shape}"
            )

        self.params = params

        # Define a solver with the sparsity values of the kernel elements as boolean variables
        self.solver = Solver()
        self._kernel_in = BitVec("kiv", params.kernel_size_in)
        self._kernel_out = BitVec("kov", params.kernel_size_out)

        # Apply the kernel priors (we won't use them again later)
        ki_set_mask = np_to_bitvec(kernel_in_prior, 2)
        ki_unset_mask = np_to_bitvec(kernel_in_prior, 0)
        ko_set_mask = np_to_bitvec(kernel_out_prior, 2)
        ko_unset_mask = np_to_bitvec(kernel_out_prior, 0)
        self.solver.add(
            (self._kernel_in & ki_set_mask) == ki_set_mask,
            (self._kernel_in & ki_unset_mask) == 0,
            (self._kernel_out & ko_set_mask) == ko_set_mask,
            (self._kernel_out & ko_unset_mask) == 0,
        )

    def add_in_out_map(self, in_len: int, in_nan_range: Tuple[int, int], out_nan_idx: np.ndarray):
        # Encode each window being corrupted as a boolean variable
        _, num_wins, _ = self.params.get_metrics_for_input(in_len)
        var_id = len(self.solver.assertions())
        corrupted_wins = BitVec(f"cw_{var_id}", num_wins)

        for win_idx, ((in_start, in_stop), (out_start, out_stop)) in enumerate(self.params.iter_kernel_map(num_wins)):
            win_mask = 1 << win_idx

            # The kernel can only output nans (=be corrupted) if it has any overlap with the input nans
            kernel_in_nan_range = (
                int(max(in_nan_range[0], in_start) - in_start),
                int(min(in_nan_range[1], in_stop) - in_start),
            )
            if (
                in_nan_range[0] < in_stop
                and in_start < in_nan_range[1]
                and kernel_in_nan_range[0] < kernel_in_nan_range[1]
            ):
                assert 0 <= kernel_in_nan_range[0] < kernel_in_nan_range[1] <= self.params.kernel_size_in
                kernel_slice = Extract(kernel_in_nan_range[1] - 1, kernel_in_nan_range[0], self._kernel_in)
                # If any bit in the slice is set, then the window is corrupted
                self.solver.add(UGT(kernel_slice, 0) == ((corrupted_wins & win_mask) != 0))
            else:
                self.solver.add(corrupted_wins & win_mask == 0)

        for out_start, out_end, overlapping_wins in self.params.get_inverse_kernel_map(in_len):
            for out_idx in range(out_start, out_end):
                # TODO: optimize
                any_corrupted_constraint = Or(
                    *[
                        And(
                            (corrupted_wins & (1 << win_idx)) != 0,
                            (self._kernel_out & (1 << (kernel_out_start + out_idx - out_start))) != 0,
                        )
                        for win_idx, kernel_out_start, kernel_out_stop in overlapping_wins
                    ]
                )
                self.solver.add(any_corrupted_constraint if out_idx in out_nan_idx else Not(any_corrupted_constraint))

    def has_solution(self) -> bool:
        # If the solver can't find any solution, then the parameters do not allow to explain the observed nans
        return self.solver.check() != unsat

    def determine(self) -> Tuple[np.ndarray | None, np.ndarray | None]:
        # TODO: doc
        if not self.has_solution():
            return None, None

        kernel_in_values = get_init_kernel_array(self.params.kernel_size_in)
        kernel_out_values = get_init_kernel_array(self.params.kernel_size_out)

        assertions = self.solver.assertions()
        for kernel_var, kernel_value_array in (
            (self._kernel_in, kernel_in_values),
            (self._kernel_out, kernel_out_values),
        ):
            for i in range(kernel_var.size()):
                assertion_to_add = None
                if self.solver.check(kernel_var & (1 << i) != 0) == unsat:
                    kernel_value_array[i] = 0
                    assertion_to_add = kernel_var & (1 << i) == 0
                elif self.solver.check(kernel_var & (1 << i) == 0) == unsat:
                    kernel_value_array[i] = 2
                    assertion_to_add = kernel_var & (1 << i) != 0

                if assertion_to_add is not None and assertion_to_add not in assertions:
                    self.solver.add(assertion_to_add)

        return kernel_in_values, kernel_out_values
