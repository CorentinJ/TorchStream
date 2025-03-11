from dataclasses import dataclass
from typing import Iterable, Tuple, Union

import numpy as np
import torch
from colorama import Fore
from torch import nn
from z3 import And, If, Int, Ints, Or, Solver, sat

# TODO: support multiple inputs/output as long as they have the same sequence length
# TODO?: handle reflect padding
#   -> Already handled?


def get_nan_range(x: Union[torch.Tensor, np.ndarray], dim: int = -1):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    dim = range(x.ndim)[dim]
    x = x.mean(axis=tuple(i for i in range(x.ndim) if i != dim))

    corrupted_idx = np.where(np.isnan(x))[0]

    if not len(corrupted_idx):
        return None, None
    return corrupted_idx[0], corrupted_idx[-1] + 1


class NoSolutionError(Exception):
    pass


@dataclass
class SlidingWindowParams:
    kernel_size_in: int
    stride_in: int
    kernel_size_out: int
    stride_out: int
    left_pad: int
    # FIXME
    right_pad: Tuple[int, ...]

    def __repr__(self):
        return (
            "SlidingWindowParams(\n"
            + f"    left_pad: {self.left_pad}, right_pad: {self.right_pad}\n"
            + f"    kernel_size_in: {self.kernel_size_in}, stride_in: {self.stride_in}\n"
            + f"    kernel_size_out: {self.kernel_size_out}, stride_out: {self.stride_out}\n"
            + ")"
        )


class SlidingWindowParamsSolver:
    """
    TODO: doc
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
        self.p_rs = []  # TODO: remove such lists? Do we really need to keep the ref?
        self.solver.add(self.p_l >= 0, self.p_l < self.k_i)
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
        # - Padding on the left is 0, and both trimming or padding could occur on the right to line up with the windows
        # - Padding on the left is >0, in which case it would be odd to trim on the right, so we only allow padding
        p_r = Int(f"p_r_{len(self.p_rs)}")
        self.p_rs.append(p_r)
        self.solver.add(
            Or(
                And(self.p_l == 0, p_r > -self.s_i),
                And(self.p_l > 0, p_r >= 0),
            ),
            p_r < self.k_i,
        )

        # Input to output size relation with the number of windows
        c = Int(f"c_{len(self.cs)}")
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
            crs, cre = Ints(f"c_{len(self.cs)}_rs{range_idx} c_{len(self.cs)}_re{range_idx}")
            self.solver.add(0 <= crs, crs <= cre, cre < c)

            self.solver.add(
                crs
                == (If(self.p_l + in_range[0] >= self.k_i, self.p_l + in_range[0] - self.k_i + 1, 0) + self.s_i - 1)
                / self.s_i
            )
            self.solver.add(out_range[0] == crs * self.s_o)

            self.solver.add(
                cre == If(self.p_l + in_range[1] >= c * self.s_i, c - 1, (self.p_l + in_range[1]) / self.s_i)
            )
            self.solver.add(out_range[1] == cre * self.s_o + self.k_o)

        # if self.solver.check() != sat:
        #     raise NoSolutionError(
        #         # TODO: better explanation for this, course for action etc...
        #         f"Adding the constraint input={in_len} -> output={out_len} made the model unsolvable."
        #     )

    def get_sols(self, max_solutions: int = 50):
        out = []
        while self.solver.check() == sat:
            model = self.solver.model()
            # print(model)

            params = SlidingWindowParams(
                model[self.k_i].as_long(),
                model[self.s_i].as_long(),
                model[self.k_o].as_long(),
                model[self.s_o].as_long(),
                model[self.p_l].as_long(),
                tuple(model[pr].as_long() for pr in self.p_rs),
            )
            out.append(params)

            self.solver.add(
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


class SimpleSlidingWindowTransform:
    def __init__(self, params: SlidingWindowParams):
        self.params = params

    def __call__(self, x: np.ndarray, right_pad):
        x = np.concatenate([np.zeros(self.params.left_pad), x])
        if right_pad < 0:
            x = x[:right_pad]
        else:
            x = np.concatenate([x, np.zeros(right_pad)])

        # TODO: make methods of sliding window params
        num_windows = (len(x) - self.params.kernel_size_in) / self.params.stride_in + 1
        assert num_windows.is_integer()
        num_windows = int(num_windows)
        output_length = (num_windows - 1) * self.params.stride_out + self.params.kernel_size_out

        out = np.zeros(output_length)
        for i in range(num_windows):
            in_sli = slice(i * self.params.stride_in, i * self.params.stride_in + self.params.kernel_size_in)
            out_sli = slice(i * self.params.stride_out, i * self.params.stride_out + self.params.kernel_size_out)
            out[out_sli] += np.mean(x[in_sli])

        return out


@torch.no_grad()
def test_conv1d():
    a = nn.Conv1d(1, 1, kernel_size=5, stride=2)
    solver = SlidingWindowParamsSolver()

    # TODO: edge cases
    in_lens = (12,)  # 14, 17)
    nan_inputs = [
        (7, 8),
    ]  # (5, 10), (11, 13)]
    out_lens = []
    out_ranges = []
    for in_len, nan_input in zip(in_lens, nan_inputs):
        inp = torch.randn(1, 1, in_len)
        inp[0, 0, slice(*nan_input)] = torch.nan

        out = a(inp)
        out_lens.append(out.size(2))

        left, right = get_nan_range(out)
        print(f"In: {in_len}, Out: {out.size(2)}, Nans: {nan_input} -> {left, right}")
        out_ranges.append((left, right))

        solver.add_all((in_len, out.size(2)), [((nan_input[0], nan_input[1]), (left, right))])

    print()
    all_params = solver.get_sols()

    n_sols = 0
    for params in all_params:
        print("--- Solution ---")
        print(params)

        failed = False
        for in_len, nan_input, out_len, out_range, rpad in zip(
            in_lens, nan_inputs, out_lens, out_ranges, params.right_pad
        ):
            a = SimpleSlidingWindowTransform(params)

            inp = np.random.randn(in_len)
            inp[nan_input[0] : nan_input[1]] = torch.nan

            out = a(inp, rpad)
            if len(out) != out_len:
                print(f"{Fore.RED}Failed: expected out len {out_len}, got {len(out)}{Fore.RESET}")
                failed = True

            left, right = get_nan_range(out)
            if (left, right) != out_range:
                print(f"{Fore.RED}Failed: expected out range {out_range}, got {(left, right)}{Fore.RESET}")
                failed = True

        if not failed:
            print(f"{Fore.GREEN}Success!{Fore.RESET}")
            n_sols += 1

    print(f"\nFound {n_sols}/{len(all_params)} working solutions")


test_conv1d()
