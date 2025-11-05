import logging
import math
from bisect import bisect_left
from typing import List, Tuple

import numpy as np
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SlidingWindowInOutRelSampler:
    def __init__(self):
        self.obs = np.ndarray(shape=(0, 2), dtype=int)

    @tracer.start_as_current_span("in_out_rel_sampler.add_in_out_size")
    def add_in_out_size(self, in_len: int, out_len: int):
        """
        TODO: doc
        """
        if in_len < 1:
            raise ValueError("The input length must be a strictly positive integer")
        if out_len < 1:
            raise ValueError(
                "The output length must be a strictly positive integer, this sampler does not model the "
                "minimum input size"
            )

        # Add the observation in sorted order
        insert_idx = bisect_left(self.obs[:, 0], in_len)
        if insert_idx < len(self.obs) and tuple(self.obs[insert_idx]) == (in_len, out_len):
            raise ValueError("This observation has already been added")
        self.obs = np.insert(self.obs, insert_idx, (in_len, out_len), axis=0)

    def _solve_so(self) -> Tuple[int | None, int | None]:
        if len(self.obs) == 0:
            raise RuntimeError("No observations have been added yet")

        diffs = self.obs[1:] - self.obs[:-1]

        # If we have two consecutive x values with different y, we're done
        direct_step = diffs[(diffs[:, 0] == 1) & (diffs[:, 1] > 0)]
        if len(direct_step) > 0:
            return int(direct_step[0, 1]), None

        # No delta y values yet? Request twice the maximum of all input sizes
        if np.all(diffs[:, 1] == 0):
            return None, 2 * int(np.max(self.obs[:, 0]))

        # TODO: more data efficient approach here

        # No solution yet? Return the mid point between two obs that had the smallest strictly positive output size
        # difference
        min_pos_out_diff = np.min(diffs[diffs[:, 1] > 0, 1])
        min_pos_out_diff_idx = np.where(diffs[:, 1] == min_pos_out_diff)[0][0]
        in_size_a = self.obs[min_pos_out_diff_idx, 0]
        in_size_b = self.obs[min_pos_out_diff_idx + 1, 0]
        return None, int((in_size_a + in_size_b) // 2)

    def _solve_si_bounds(self, s_o: int) -> Tuple[Tuple[int, int] | None, int | None]:
        s_i_bounds = [1, float("inf")]

        # Take all observations pairwise to infer s_i bounds
        for obs1_idx in range(len(self.obs)):
            for obs2_idx in range(obs1_idx + 1, len(self.obs)):
                in_diff = self.obs[obs2_idx, 0] - self.obs[obs1_idx, 0]
                quotient_diff = (self.obs[obs2_idx, 1] - self.obs[obs1_idx, 1]) // s_o

                s_i_bounds[0] = max(s_i_bounds[0], int(math.ceil((in_diff + 2) / (quotient_diff + 2))))
                if quotient_diff > 1:
                    s_i_bounds[1] = min(s_i_bounds[1], int(math.ceil((in_diff - 1) / (quotient_diff - 1))))

        # If we have no upper bound we need more data, request double the maximum input size again
        if s_i_bounds[1] == float("inf"):
            return None, 2 * int(np.max(self.obs[:, 0]))

        return tuple(s_i_bounds), None

    def _verify_parameters(self, s_i: int, s_o: int, isbc: int, osbc: int) -> bool:
        expected_out_len = ((self.obs[:, 0] + isbc) // s_i) * s_o + osbc
        return np.all(expected_out_len == self.obs[:, 1])

    def _determine_unique_solution(
        self, s_i_bounds: Tuple[int, int], s_o: int
    ) -> Tuple[Tuple[int, int, int, int] | None, int | None]:
        # At this point in the process, we only have a small finite number of solutions possible. We can brute force the
        # remaining parameters.
        possible_params = []
        for s_i in range(s_i_bounds[0], s_i_bounds[1] + 1):
            for isbc in range(s_i):
                osbc = int(self.obs[0, 1] - ((self.obs[0, 0] + isbc) // s_i) * s_o)
                if self._verify_parameters(s_i, s_o, isbc, osbc):
                    possible_params.append((s_i, s_o, isbc, osbc))

        if len(possible_params) == 1:
            return possible_params[0], None

        # If our solution isn't unique, we'll work with the max infogain. We'll stick to the minimum input size found
        # so far to avoid going below the transform's minimum input size.
        in_to_out_sizes = compute_in_to_out_sizes(possible_params)
        min_in_size = int(np.min(self.obs[:, 0]))
        max_infogain_input_size = input_size_by_max_infogain(in_to_out_sizes[:, min_in_size:]) + min_in_size
        return None, max_infogain_input_size

    @tracer.start_as_current_span("in_out_rel_sampler.solve")
    def solve(self) -> Tuple[Tuple[int, int, int, int] | None, int | None]:
        s_o, next_in_size = self._solve_so()
        if next_in_size:
            return None, next_in_size

        s_i_bounds, next_in_size = self._solve_si_bounds(s_o)
        if next_in_size:
            return None, next_in_size

        return self._determine_unique_solution(s_i_bounds, s_o)


def compute_in_to_out_sizes(
    shape_params: List[tuple],
    max_input_size=10_000,
) -> np.ndarray:
    # TODO: doc
    si, so, isbc, osbc = [np.array(param_group)[..., None] for param_group in zip(*shape_params)]

    out_sizes = np.stack([np.arange(0, max_input_size)] * len(shape_params))
    out_sizes = np.maximum(((out_sizes + isbc) // si) * so + osbc, 0)

    return out_sizes


@tracer.start_as_current_span("input_size_by_max_infogain")
def input_size_by_max_infogain(in_to_out_sizes: np.ndarray, method="entropy") -> int:
    # TODO: doc
    if in_to_out_sizes.shape[0] <= 2 or method == "n_unique":
        # Vectorized method for counting unique values
        unique_counts = 1 + np.count_nonzero(np.diff(np.sort(in_to_out_sizes, axis=0), axis=0), axis=0)
        in_size = int(np.argmax(unique_counts[1:])) + 1
        assert unique_counts[in_size] > 1

    elif method == "entropy":
        # Vectorized entropy computation
        R, C = in_to_out_sizes.shape
        s = np.sort(in_to_out_sizes, axis=0)
        sf = s.ravel(order="F")

        breaks = np.empty(R * C, dtype=bool)
        breaks[0] = True
        at_col_start = np.zeros(R * C, dtype=bool)
        at_col_start[::R] = True

        same = sf[1:] == sf[:-1]
        breaks[1:] = at_col_start[1:] | (~same)

        start = np.flatnonzero(breaks)
        end = np.r_[start[1:], R * C]
        lens = end - start
        cols = start // R

        H = np.zeros(C, dtype=float)
        p = lens.astype(float) / float(R)
        np.add.at(H, cols, -(p * np.log2(p)))

        in_size = int(np.argmax(H[1:])) + 1
        assert H[in_size] > 0

    else:
        raise ValueError(f"Unknown method '{method}'")

    return in_size
