import logging
from typing import Tuple

import numpy as np
from z3 import And, Bool, Not, Or, unsat

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


# TODO: merge with _get_window_corruption_map, it's the same logic
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

    for (in_start, in_stop), (out_start, out_stop) in params.iter_kernel_map(num_wins=num_wins):
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
        self._kernel_in = kernel_in_prior.copy()
        self._kernel_out = kernel_out_prior.copy()
        self._solvable = True

        # # Define a solver with the sparsity values of the kernel elements as boolean variables
        # self.solver = Solver()
        # self._kernel_in_var = [Bool("kernel_in_" + str(i)) for i in range(self.params.kernel_size_in)]
        # self._kernel_out_var = [Bool("kernel_out_" + str(i)) for i in range(self.params.kernel_size_out)]

        # # Apply the kernel priors
        # # FIXME!
        # for idx, val in enumerate(kernel_in_prior):
        #     if val == 0:
        #         self.solver.add(Not(self._kernel_in_var[idx]))
        #     elif val == 2:
        #         self.solver.add(self._kernel_in_var[idx])
        # for idx, val in enumerate(kernel_out_prior):
        #     if val == 0:
        #         self.solver.add(Not(self._kernel_out_var[idx]))
        #     elif val == 2:
        #         self.solver.add(self._kernel_out_var[idx])

    def _get_window_corruption_map(
        self, in_len: int, in_nan_range: Tuple[int, int], out_nan_idx: np.ndarray
    ) -> np.ndarray | None:
        """
        Given an input and output nan map, returns the window corruption array (with values 0, 1, 2) based on the
        current kernel sparsity assumptions. Returns None if the input and output NaNs observed cannot be reconciled
        with the current parameters or kernel sparsity assumptions.
        """
        padding, num_wins, out_len = self.params.get_metrics_for_input(in_len)

        in_vec = np.zeros(in_len, dtype=bool)
        in_vec[in_nan_range[0] : in_nan_range[1]] = 1
        padded_in_vec = np.pad(in_vec, padding)
        # TODO? rewrite with scipy.ndimage.generic_filter? Probably not useful perf wise
        k_i, s_i = self.params.kernel_size_in, self.params.stride_in
        corrupted_wins = np.array(
            [np.max(self._kernel_in * padded_in_vec[i * s_i : i * s_i + k_i]) for i in range(num_wins)]
        )

        out_vec = np.zeros(out_len, dtype=bool)
        out_vec[out_nan_idx] = 1
        for out_start, out_end, overlapping_wins in self.params.get_inverse_kernel_map(in_len):
            out_slice = out_vec[out_start:out_end].copy()

            # Partition windows depending on whether they must have corrupted the output, or might have
            known_corr = np.zeros_like(out_slice, dtype=int)
            maybe_corr = np.zeros_like(out_slice, dtype=int)
            maybe_corr_win_idx = []
            for win_idx, kernel_out_start, kernel_out_stop in overlapping_wins:
                if corrupted_wins[win_idx] == 2:
                    known_corr = np.maximum(known_corr, self._kernel_out[kernel_out_start:kernel_out_stop])
                elif corrupted_wins[win_idx] == 1:
                    maybe_corr = np.maximum(maybe_corr, self._kernel_out[kernel_out_start:kernel_out_stop])
                    maybe_corr_win_idx.append(win_idx)

            # Ensure the ouput matches the known corrupted windows
            # NOTE: this is the converse implication operator i.e. if known_corr[i] is 2, then out_slice[i] must be True
            if not np.all(out_slice ** (known_corr == 2)):
                return None

            # We'll explain the nans by the know corrupted windows, even if some elements are unsure
            out_slice = np.logical_and(out_slice, known_corr == 0)

            # If we can't explain the remaining nans with the maybe corrupted windows, then there's no solution
            if np.logical_and(out_slice, maybe_corr == 0).any():
                return None

            # If we only have one window left that might have corrupted the output, we can deduce its state
            # NOTE: This is technically suboptimal because we're scanning the output left to right and might miss
            # a case where knowing a future window is corrupted would allow to deduce the value of a past window,
            # but who cares this is a heuristic to crunch down the solver's work, not a full solution.
            if len(maybe_corr_win_idx) == 1:
                corrupted_wins[maybe_corr_win_idx[0]] = 2

        return corrupted_wins

    def add_in_out_map(self, in_len: int, in_nan_range: Tuple[int, int], out_nan_idx: np.ndarray):
        if not self._solvable:
            return

        expected_out = get_nan_map(
            self.params,
            in_len,
            in_nan_range,
            kernel_in=self._kernel_in,
            kernel_out=self._kernel_out,
        )

        for idx, value in enumerate(expected_out):
            if (value == 2 and idx not in out_nan_idx) or (value == 0 and idx in out_nan_idx):
                self._solvable = False
                return

        return

        # FIXME!
        ###########

        # We'll save some work for the solver by figuring out which windows are corrupted based on the current kernel
        # assumptions, and only write constraints for those windows.
        corrupted_wins = self._get_window_corruption_map(in_len, in_nan_range, out_nan_idx)
        self._solvable &= corrupted_wins is not None
        if not self._solvable:
            return

        # win_map = list(self.params.iter_kernel_map(in_len=in_len))
        padding, num_wins, out_len = self.params.get_metrics_for_input(in_len)

        # Encode each window being corrupted as a boolean variable
        var_id = len(self.solver.assertions())
        corrupted_win_vars = {
            i: Bool(f"corrupted_win_{var_id}_{i}") for i, value in enumerate(corrupted_wins) if value == 1
        }

        for win_idx, ((in_start, in_stop), (out_start, out_stop)) in enumerate(win_map):
            # The kernel can only output nans (=be corrupted) if it has any overlap with the input nans
            kernel_in_nan_range = (
                max(in_nan_range[0], in_start) - in_start,
                min(in_nan_range[1], in_stop) - in_start,
            )
            if (
                in_nan_range[0] < in_stop
                and in_start < in_nan_range[1]
                and kernel_in_nan_range[0] < kernel_in_nan_range[1]
            ):
                assert 0 <= kernel_in_nan_range[0] < kernel_in_nan_range[1] <= self.params.kernel_size_in
                kernel_slice = Extract(kernel_in_nan_range[1] - 1, kernel_in_nan_range[0], self._kernel_in_var)
                # If any bit in the slice is set, then the window is corrupted
                self.solver.add(UGT(kernel_slice, 0) == ((corrupted_wins & win_mask) != 0))
            else:
                self.solver.add(corrupted_wins & win_mask == 0)

        for out_start, out_end, overlapping_wins in self.params.get_inverse_kernel_map(in_len):
            for out_idx in range(out_start, out_end):
                any_corrupted_constraint = Or(
                    *[
                        And(corrupted_wins[win_idx], self._kernel_out_var[kernel_out_start + out_idx - out_start])
                        for win_idx, kernel_out_start, kernel_out_stop in overlapping_wins
                    ]
                )
                self.solver.add(any_corrupted_constraint if out_idx in out_nan_idx else Not(any_corrupted_constraint))

    def has_solution(self) -> bool:
        # If the solver can't find any solution, then the parameters do not allow to explain the observed nans
        # FIXME!!
        return self._solvable  # and (self.solver.check() != unsat)

    def determine(self) -> Tuple[np.ndarray | None, np.ndarray | None]:
        # TODO: doc
        if not self.has_solution():
            return None, None

        kernel_in_values = get_init_kernel_array(self.params.kernel_size_in)
        kernel_out_values = get_init_kernel_array(self.params.kernel_size_out)

        assertions = self.solver.assertions()
        for kernel_var, kernel_value_array in (
            (self._kernel_in_var, kernel_in_values),
            (self._kernel_out_var, kernel_out_values),
        ):
            for i in range(len(kernel_var)):
                assertion_to_add = None
                if self.solver.check(kernel_var[i]) == unsat:
                    kernel_value_array[i] = 0
                    assertion_to_add = Not(kernel_var[i])
                elif self.solver.check(Not(kernel_var[i])) == unsat:
                    kernel_value_array[i] = 2
                    assertion_to_add = kernel_var[i]

                if assertion_to_add is not None and assertion_to_add not in assertions:
                    self.solver.add(assertion_to_add)

        return kernel_in_values, kernel_out_values
