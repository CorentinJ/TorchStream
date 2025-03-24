import math
from collections import Counter
from copy import deepcopy
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
from colorama import Fore
from torch import nn
from z3 import If, Int, Ints, Or, Solver, sat

from torchstream.sliding_window.nan_trick import check_nan_trick, get_nan_range, set_nan_range
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.tensor_provider import TensorProvider, TensorSpec


class NoSolutionError(Exception):
    pass


class SlidingWindowParamsSolver:
    """
    TODO: doc
    TODO: sort these out
    - The input size to the number of windows is a deterministic, stepwise monotonic linear function.

    TODO: padding
    - The most common padding is constant, with zeroes. But there are other padding methods such as reflect or
    circular that sample from the input array to determine the padding values. How does this compromise results from
    the nan trick? Does the transform still qualify as a sliding window approach (e.g. if it reads in the middle of
    the vector to get padding values)?
    """

    def __init__(self):
        # NOTE: possible constructor parameters:
        # - Left padding variable

        # Define the parameters we're trying to uniquely determine
        self.solver = Solver()
        # k_i and s_i are respectively the input kernel size and stride (NOTE: it's technically the kernel span,
        # i.e. the whole span of the kernel if dilation > 1)
        self.k_i, self.s_i = Ints("k_i s_i")
        # It would be highly unusual to have a stride larger than the kernel size, leading to inputs being dropped.
        self.solver.add(self.k_i >= self.s_i, self.s_i > 0)
        # k_o and s_o are respectively the output kernel size and stride. These are both 1 for normal convolutions,
        # but not for transposed convolutions.
        self.k_o, self.s_o = Ints("k_o s_o")
        # Again, it would be strange to have a stride larger than the kernel size, leading to gaps in the output.
        self.solver.add(self.k_o >= self.s_o, self.s_o > 0)
        # The left input padding. I have not yet seen a case where varying the left padding is useful, so we'll
        # assume it constant. Also, there is no point in making the padding higher than the kernel size, as it would
        # waste compute on constant values.
        self.p_l = Int("p_l")
        self.solver.add(0 <= self.p_l, self.p_l < self.k_i)
        # TODO doc
        self.p_r = Int("p_r")
        self.solver.add(0 <= self.p_r, self.p_r < self.k_i)

        # Number of sliding windows for constraints
        self.cs = []

    # TODO: name
    def add_all(self, in_out_len: Tuple[int, int], in_out_ranges: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]):
        """
        TODO: doc
        """
        in_len, out_len = int(in_out_len[0]), int(in_out_len[1])
        if in_len < 1 or out_len < 1:
            raise ValueError("Input and output lengths must be strictly positive integers")

        # Input to output size relation with the number of windows
        c_idx = len(self.cs)
        c = Int(f"c_{c_idx}")
        self.cs.append(c)
        self.solver.add(
            # We necessarily have at least one window (empty outputs are not allowed for this solver)
            c > 0,
            # Padding
            c == (self.p_l + in_len + self.p_r - self.k_i) / self.s_i + 1,
            out_len == (c - 1) * self.s_o + self.k_o,
        )

        for range_idx, (in_range, out_range) in enumerate(in_out_ranges):
            in_range = (int(in_range[0]), int(in_range[1]))
            out_range = (int(out_range[0]), int(out_range[1]))
            if in_range[0] < 0 or out_range[0] < 0:
                raise ValueError("Input and output ranges must be non-negative integers")
            if in_range[1] <= in_range[0] or out_range[1] <= out_range[0]:
                raise ValueError("Input and output ranges must be non-empty")

            # The start of both the input and the output range correspond to the same window. The same can be said
            # for the end of the ranges.
            # FIXME: notation difference: c above is the number of windows, cs and ce are window indices
            crs, cre = Ints(f"c_{c_idx}_rs{range_idx} c_{c_idx}_re{range_idx}")
            self.solver.add(0 <= crs, crs <= cre, cre < c)

            self.solver.add(
                crs
                == (If(self.p_l + in_range[0] >= self.k_i, self.p_l + in_range[0] - self.k_i + 1, 0) + self.s_i - 1)
                / self.s_i
            )
            self.solver.add(out_range[0] == crs * self.s_o)

            self.solver.add(
                cre == If(self.p_l + in_range[1] > (c - 1) * self.s_i, c - 1, (self.p_l + in_range[1] - 1) / self.s_i)
            )
            self.solver.add(out_range[1] == cre * self.s_o + self.k_o)

        # if self.solver.check() != sat:
        #     raise NoSolutionError(
        #         # TODO: better explanation for this, course for action etc...
        #         f"Adding the constraint input={in_len} -> output={out_len} made the model unsolvable."
        #     )

    def get_sols(self, max_solutions: int = 50):
        # TODO: use push/pop instead?
        solver = deepcopy(self.solver)

        out = []
        while solver.check() == sat:
            model = solver.model()
            # print(model)

            params = SlidingWindowParams(
                kernel_size_in=model[self.k_i].as_long(),
                stride_in=model[self.s_i].as_long(),
                left_pad=model[self.p_l].as_long(),
                right_pad=model[self.p_r].as_long(),
                kernel_size_out=model[self.k_o].as_long(),
                stride_out=model[self.s_o].as_long(),
            )
            out.append(params)

            solver.add(
                Or(
                    self.k_i != model[self.k_i],
                    self.s_i != model[self.s_i],
                    self.p_l != model[self.p_l],
                    self.p_r != model[self.p_r],
                    self.k_o != model[self.k_o],
                    self.s_o != model[self.s_o],
                )
            )

            if len(out) >= max_solutions:
                break

        return out


def _count_unique_arrays(
    arrays: List[np.ndarray], counts: Iterable[int] | None = None
) -> Tuple[List[np.ndarray], List[int]]:
    if counts is None:
        counts = [1] * len(arrays)
    counts = list(counts)
    if len(counts) != len(arrays):
        raise ValueError()

    # NOTE: this can also be achieved with np.unique given an axis arg, but it requires padding arrays to the same
    # shape...
    array_idx_map = {}
    unique_arrays = []
    unique_arrays_count = []
    for array in arrays:
        key = array.tobytes()

        if key not in array_idx_map:
            array_idx_map[key] = len(unique_arrays)
            unique_arrays.append(array.copy())
            unique_arrays_count.append(0)
        array_idx = array_idx_map[key]

        unique_arrays_count[array_idx] += 1

    return unique_arrays, unique_arrays_count


def find_nan_trick_params_by_infogain(hypotheses: SlidingWindowParams):
    # TODO: doc
    # TODO: handle cases where the input is too small?
    min_seq_size = max(sol.get_min_input_size() for sol in hypotheses)
    max_seq_size = min_seq_size + max(sol.kernel_size_in for sol in hypotheses)

    best_infogain = 0.0
    best_parameters = (None, None)
    for seq_size in range(min_seq_size, max_seq_size + 1):
        inv_maps = []
        for hypothesis in hypotheses:
            inv_map = hypothesis.get_inverse_map(seq_size)

            # FIXME
            inv_map = np.maximum(inv_map, 0)
            inv_map = np.minimum(inv_map, seq_size)

            inv_maps.append(inv_map)

        inv_maps, inv_maps_count = _count_unique_arrays(inv_maps)

        # TODO: sublinear algo
        for in_nan_idx in range(seq_size):
            # TODO: algo with nan ranges larger than 1
            in_nan_range = (in_nan_idx, in_nan_idx + 1)
            outcomes = Counter()
            for inv_map, inv_map_count in zip(inv_maps, inv_maps_count):
                out_nan_range = (
                    np.searchsorted(inv_map[:, 1], in_nan_range[0], side="right"),
                    np.searchsorted(inv_map[:, 0], in_nan_range[1], side="left"),
                )
                if out_nan_range[1] <= out_nan_range[0]:
                    out_nan_range = None
                outcomes[len(inv_map), out_nan_range] += inv_map_count

                # TODO: empirical verification

            infogain = math.log(len(hypotheses))
            for outcome_count in outcomes.values():
                infogain -= (outcome_count * math.log(outcome_count)) / len(hypotheses)

            if infogain > best_infogain:
                best_infogain = infogain
                best_parameters = (seq_size, in_nan_range)

            # Break early if the solution is optimal
            if len(outcomes) == len(hypotheses):
                return best_parameters

    return best_parameters


# TODO: allow transforms with multiple sequential inputs
#   -> Or simply call the function multiple times? unsure
@torch.no_grad()
def find_sliding_window_params_for_transform(
    trsfm: Callable,
    input_provider: TensorProvider,
    min_seq_size: int = 1,
    max_seq_size: int = None,
) -> SlidingWindowParams:
    solver = SlidingWindowParamsSolver()

    history = []
    sols = []
    while True:
        # Determine an input size
        if not sols:
            # TODO: min input size
            seq_size = 10
            in_nan_range = (5, 6)
        else:
            seq_size, in_nan_range = find_nan_trick_params_by_infogain(sols)
            if seq_size is None:
                print("Cannot differentiate between params")
                break

        # TODO: integrate in parameter search space
        seq_size = max(min_seq_size, seq_size)
        if max_seq_size:
            seq_size = min(seq_size, max_seq_size)

        print(seq_size, in_nan_range)

        # TODO: nan range lims

        x = input_provider.get_tensor(seq_size)
        # TODO: move to TensorProvider?
        assert x.size(input_provider.dim) == seq_size

        set_nan_range(x, in_nan_range, dim=input_provider.dim)

        try:
            y = trsfm(x)
        except Exception as e:
            y = e

        # FIXME: output format
        if torch.is_tensor(y):
            # FIXME: dim
            out_nan_range = get_nan_range(y, dim=-1)
            solver.add_all(
                (seq_size, y.size(-1)),
                [(in_nan_range, out_nan_range)],
            )
            print(f"In: {seq_size}, Out: {y.size(-1)}, Nans: {in_nan_range} -> {out_nan_range}")

        history.append((x, y))
        sols = solver.get_sols()

        # Nan trick verification (TODO: remove?)
        n_sols = 0
        for params in sols:
            failed = False
            success, reason = check_nan_trick(params, seq_size, y.size(-1), in_nan_range, out_nan_range)
            if not success:
                print("--- Solution ---")
                print(params)
                print(f"{Fore.RED}Failed!{Fore.RESET} Reason: {reason}")
                failed = True

            if not failed:
                n_sols += 1
        print(f"\nFound {n_sols}/{len(sols)} working solutions")

        if len(sols) <= 10:
            for sol in sols:
                print(sol)

        if len(sols) == 1:
            break

        print("----------")


temp = nn.Conv1d(1, 1, kernel_size=7, stride=2, padding=0)


def my_transform(x):
    x = torch.nn.functional.pad(x, (1, 3))
    return temp(x)


def test_conv1d():
    # conv = nn.Conv1d(1, 1, kernel_size=7, stride=1, padding=2)
    sols = find_sliding_window_params_for_transform(my_transform, TensorSpec(shape=(1, 1, -1)), max_seq_size=200)
    print(sols)


test_conv1d()
