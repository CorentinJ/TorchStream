from copy import deepcopy
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from z3 import And, Bool, If, Int, Ints, Or, Solver, sat

from torchstream.sliding_window.nan_trick import get_nan_range, set_nan_range
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.tensor_provider import TensorProvider, TensorSpec


class NoSolutionError(Exception):
    pass


class SlidingWindowParamsSolver:
    """
    TODO: doc
    The input size to the number of windows is monotonic and deterministic
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
        # The left input padding. The right padding is susceptible to varying in practice (we account for that below).
        # I have not yet seen a case where varying the left padding is useful, so we'll assume it constant.
        # There is no point in making the padding higher than the kernel size, as it would waste compute on constant
        # values.
        self.p_l = Int("p_l")
        self.solver.add(self.p_l < self.k_i)
        self.drop_right = Bool("right_drop")
        self.solver.add(If(self.drop_right, self.p_l == 0, self.p_l >= 0))

        # Right padding variables
        self.p_rs = []
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

        # Variable right padding. Two sensible cases:
        # - We drop excess inputs on the right, so there can't be right padding
        # - We pad if windows don't line up with the end of the sequence
        p_r = Int(f"p_r_{len(self.p_rs)}")
        self.p_rs.append(p_r)
        self.solver.add(
            If(self.drop_right, And(p_r <= 0, p_r > -self.s_i), p_r >= 0),
            p_r < self.k_i,
        )

        # Input to output size relation with the number of windows
        c_idx = len(self.cs)
        c = Int(f"c_{c_idx}")
        self.cs.append(c)
        self.solver.add(c > 0)
        self.solver.add(in_len + self.p_l + p_r - self.k_i == (c - 1) * self.s_i)
        self.solver.add(out_len == (c - 1) * self.s_o + self.k_o)

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
                model[self.k_i].as_long(),
                model[self.s_i].as_long(),
                model[self.k_o].as_long(),
                model[self.s_o].as_long(),
                model[self.p_l].as_long(),
                False,
                # tuple(model[pr].as_long() for pr in self.p_rs),
            )
            out.append(params)

            solver.add(
                Or(
                    self.k_i != model[self.k_i],
                    self.k_o != model[self.k_o],
                    self.s_i != model[self.s_i],
                    self.s_o != model[self.s_o],
                    self.p_l != model[self.p_l],
                )
            )

            if len(out) >= max_solutions:
                break

        return out


# TODO: allow transforms with multiple sequential inputs
@torch.no_grad()
def find_sliding_window_params_for_transform(
    trsfm: Callable,
    input_provider: TensorProvider,
    min_seq_size: int = 1,
    max_seq_size: int = None,
) -> SlidingWindowParams:
    solver = SlidingWindowParamsSolver()

    history = []
    while True:
        # Determine an input size
        # if not history:
        #     seq_size = 10
        #     in_nan_range = (5, 6)
        # else:

        # FIXME!
        seq_size = (80, 120, 120, 120, 200, 2)[len(history)]
        in_nan_range = [(0, 1), (8, 50), (9, 51), (10, 48), (199, 200), (1, 2)][len(history)]

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
        print(f"Num sols: {len(sols)}")

        if len(sols) == 1:
            break

    # n_sols = 0
    # for params in all_params:
    #     print("--- Solution ---")
    #     print(params)

    #     failed = False
    #     for in_len, nan_input, out_len, out_range, rpad in zip(
    #         in_lens, nan_inputs, out_lens, out_ranges, params.right_pad
    #     ):
    #         success, reason = check_nan_trick(params, in_len, out_len, nan_input, out_range)
    #         if not success:
    #             print(f"{Fore.RED}Failed!{Fore.RESET} Reason: {reason}")
    #             failed = True

    #     if not failed:
    #         print(f"{Fore.GREEN}Success!{Fore.RESET}")
    #         n_sols += 1

    # print(f"\nFound {n_sols}/{len(all_params)} working solutions")


def test_conv1d():
    conv = nn.Conv1d(1, 1, kernel_size=5, stride=2)
    sols = find_sliding_window_params_for_transform(conv, TensorSpec(shape=(1, 1, -1)), max_seq_size=200)
    print(sols)


test_conv1d()
