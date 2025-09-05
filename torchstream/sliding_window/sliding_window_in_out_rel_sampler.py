import logging
from typing import List

import numpy as np
from z3 import And, Int, Ints, Not, Or, Solver, sat

logger = logging.getLogger(__name__)


class SlidingWindowInOutRelSampler:
    def __init__(self):
        # Input and output strides of the sliding window.
        self.s_i, self.s_o = Ints("s_i s_o")
        # Input and output size biases in computation, canonicalized to ensure uniqueness of the relation
        self.isbc, self.osbc = Ints("isbc osbc")

        self.optimizer = Solver()
        self.optimizer.add(
            self.s_i > 0,
            self.s_o > 0,
            self.isbc >= 0,
            self.isbc < self.s_i,
            # osbc is the only parameter that can be negative -> no constraint here
        )

        # Indicates if more solutions are available
        self.exhausted = False

    # FIXME: name
    def add_in_out_size(self, in_len: int, out_len: int):
        """
        TODO: doc
        """
        if in_len < 1:
            raise ValueError("The input length must be a strictly positive integer")
        if out_len < 0:
            raise ValueError("The output length must be a non-negative integer")

        # z3 efficient encoding of out_len = max(0, ((in_len + isbc) // s_i) * s_o + osbc)
        quotient = Int(f"quotient_{in_len}_{out_len}")
        self.optimizer.add(quotient >= 0, quotient <= in_len + self.isbc)
        self.optimizer.add(self.s_i * quotient <= in_len + self.isbc)
        self.optimizer.add(in_len + self.isbc < self.s_i * (quotient + 1))

        out_len_var = self.s_o * quotient + self.osbc
        self.optimizer.add((out_len_var <= 0) if out_len == 0 else (out_len_var == out_len))

    def get_new_solutions(self, known_sols: List, max_sols=2):
        # TODO: doc
        out_sols = list(known_sols)

        # Exclude previous solutions from the search
        self.optimizer.push()
        for sol in out_sols:
            self.optimizer.add(
                Not(And(self.s_i == sol[0], self.s_o == sol[1], self.isbc == sol[2], self.osbc == sol[3]))
            )

        # Search for newer solutions
        while len(out_sols) < max_sols:
            if self.optimizer.check() == sat:
                model = self.optimizer.model()
                model_values = (
                    model[self.s_i].as_long(),
                    model[self.s_o].as_long(),
                    model[self.isbc].as_long(),
                    model[self.osbc].as_long(),
                )
                out_sols.append(model_values)

                # Enforce new solutions only
                new_sol_constraint = Or(
                    self.s_i != model[self.s_i],
                    self.s_o != model[self.s_o],
                    self.isbc != model[self.isbc],
                    self.osbc != model[self.osbc],
                )
                self.optimizer.add(new_sol_constraint)
            else:
                self.exhausted = True
                break
        self.optimizer.pop()

        return out_sols


def most_discriminative_input_size(
    shape_params: list[tuple],
    max_input_size=10_000,
    method="entropy",
) -> tuple[int, np.ndarray]:
    si, so, isbc, osbc = [np.array(param_group)[..., None] for param_group in zip(*shape_params)]

    out_sizes = np.stack([np.arange(0, max_input_size)] * len(shape_params))
    out_sizes = np.maximum(((out_sizes + isbc) // si) * so + osbc, 0)

    if method == "n_unique":
        # Vectorized method for counting unique values
        unique_counts = 1 + np.count_nonzero(np.diff(np.sort(out_sizes, axis=0), axis=0), axis=0)
        in_size = int(np.argmax(unique_counts[1:])) + 1
        assert unique_counts[in_size] > 1

    elif method == "entropy":
        # Vectorized entropy computation
        R, C = out_sizes.shape
        s = np.sort(out_sizes, axis=0)
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

    return in_size, out_sizes[:, in_size]

    # NOTE: I started writing this more efficient version but then I realized the above clocks at 1ms, and we're not
    # going to use input sizes that are magnitudes larger than 10^4 anyway.

    # k_min = np.maximum(0, np.ceil(-osbc / so).astype(int))
    # x_min = k_min * si - isbc
    # x_start = max(1, np.min(x_min))

    # slopes = so / si
    # c_mat = np.zeros((len(shape_params), len(shape_params)))
    # b_mat = np.zeros_like(c_mat, dtype=int)
    # for i, j in itertools.combinations(range(len(shape_params)), 2):
    #     if slopes[i] != slopes[j]:
    #         c_mat[i, j] = (so[i] * isbc[i]) / si[i] + osbc[i] - (so[j] * isbc[j]) / si[j] - osbc[j]
    #         b_mat[i, j] = np.ceil(
    #             (np.abs(c_mat[i, j]) + so[i] + so[j] + 1) / np.abs(slopes[i] - slopes[j])
    #         ).astype(int)

    # import matplotlib.pyplot as plt

    # plt.plot(unique_counts)
    # plt.vlines(
    #     [x_start, np.max(x_min), in_size, np.max(b_mat)],
    #     0,
    #     len(shape_params),
    #     colors=["green", "orange", "red", "blue"],
    # )
    # logger.info(x_min)
    # logger.info(b_mat)
    # logger.info(c_mat)
    # plt.show()
