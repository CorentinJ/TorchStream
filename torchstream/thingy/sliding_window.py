from typing import Tuple

import numpy as np
import torch
from torch import nn
from z3 import And, Int, Ints, Or, Solver, sat

# TODO: support multiple inputs/output as long as they have the same sequence length


def printsol(model):
    var_dict = {str(decl): model[decl] for decl in model.decls()}
    print("--- Solution ---")
    var_right_pad_names = [var for var in var_dict if var.startswith("p_r_")]
    # FIXME: alphanum
    var_right_pad_vals = [str(var_dict[var]) for var in sorted(var_right_pad_names)]
    print(f"Padding: left={var_dict['p_l']}, right=({', '.join(var_right_pad_vals)})")
    num_win_names = [var for var in var_dict if var.startswith("c_") and not var.startswith("c_reg_")]
    num_win_vals = [str(var_dict[var]) for var in sorted(num_win_names)]
    print(f"Num windows: ({', '.join(num_win_vals)})")
    print(f"Input: kernel_size={var_dict['k_i']}, stride={var_dict['s_i']}")
    print(f"Output: kernel_size={var_dict['k_o']}, stride={var_dict['s_o']}")
    print()
    # print(model)


class NoSolutionError(Exception):
    pass


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

    def add_input_to_output_length(self, input_length: int, output_length: int):
        """
        TODO: doc
        """
        input_length = int(input_length)
        output_length = int(output_length)
        if input_length < 1 or output_length < 1:
            raise ValueError("Input and output lengths must be strictly positive integers")

        # Variable right padding. Two sensible cases:
        # - Padding on the left is 0, and both trimming or padding could occur on the right to line up with the windows
        # - Padding on the left is >0, in which case it would be odd to trim on the right, so we only allow padding
        p_r = Int(f"p_r_{len(self.p_rs)}")
        self.p_rs.append(p_r)
        self.solver.add(
            Or(
                And(self.p_l == 0, p_r > -self.k_i),
                And(self.p_l > 0, p_r >= 0),
            ),
            p_r < self.k_i,
        )

        # Input to output size relation
        c = Int(f"c_{len(self.cs)}")
        self.cs.append(c)
        self.solver.add(c > 0)
        self.solver.add(input_length + self.p_l + p_r - self.k_i == (c - 1) * self.s_i)
        self.solver.add(output_length == (c - 1) * self.s_o + self.k_o)

        if self.solver.check() != sat:
            raise NoSolutionError(
                # TODO: better explanation for this, course for action etc...
                f"Adding the constraint input={input_length} -> output={output_length} made the model unsolvable."
            )

    def add_input_to_output_range(self, input_range: Tuple[int, int], output_range: Tuple[int, int]):
        """
        TODO: doc
        """
        input_range = (int(input_range[0]), int(input_range[1]))
        output_range = (int(output_range[0]), int(output_range[1]))
        if input_range[0] < 0 or output_range[0] < 0:
            raise ValueError("Input and output ranges must be non-negative integers")
        if input_range[1] <= input_range[0] or output_range[1] <= output_range[0]:
            raise ValueError("Input and output ranges must be non-empty")

        # The number of windows that overlap with the input range
        c = Int(f"c_reg_{len(self.cs)}")
        self.cs.append(c)

        # The number of windows that overlap with the input range either be the ceil or floor of the division
        # below (z3 uses floor division on ints)
        in_len = input_range[1] - input_range[0]
        self.solver.add(
            Or(
                c == (in_len + self.k_i - 1) / self.s_i,
                c == ((in_len + self.k_i - 1) + self.s_i - 1) / self.s_i,
            )
        )
        # From the number of windows, we uniquely determine the region output size
        out_len = output_range[1] - output_range[0]
        self.solver.add(out_len == (c - 1) * self.s_o + self.k_o)

        # We can also make deductions based on the start position in the input vs. the output
        if output_range[0] == 0:
            self.solver.add(input_range[0] + self.p_l <= self.k_i)
        else:
            self.solver.add(output_range[0] == ((input_range[0] - self.k_i + 1) + self.s_i - 1) / self.s_i)

        # TODO (?): constraint on end pos relative to padding

        if self.solver.check() != sat:
            raise NoSolutionError(
                # TODO: better explanation for this, course for action etc...
                f"Adding the constraint input={input_range} -> output={output_range} made the model unsolvable."
            )

    def get_sol(self):
        # if self.solver.check() == sat:
        #     model1 = self.solver.model()
        #     printsol(model1)
        #     self.solver.add(
        #         Or(
        #             self.k_i != model1[self.k_i],
        #             self.k_o != model1[self.k_o],
        #             self.s_i != model1[self.s_i],
        #             self.s_o != model1[self.s_o],
        #         )
        #     )
        #     if self.solver.check() == sat:
        #         printsol(self.solver.model())
        #     else:
        #         print("The solution is unique.")
        # else:
        #     print("No solution")

        while self.solver.check() == sat:
            model1 = self.solver.model()
            printsol(model1)
            self.solver.add(
                Or(
                    self.k_i != model1[self.k_i],
                    self.k_o != model1[self.k_o],
                    self.s_i != model1[self.s_i],
                    self.s_o != model1[self.s_o],
                )
            )


@torch.no_grad()
def test_conv1d():
    a = nn.Conv1d(1, 1, kernel_size=5, stride=2)
    solver = SlidingWindowParamsSolver()

    in_lens = (12,)  # 14, 17)
    nan_inputs = [
        (7, 8),
    ]  # (5, 10), (11, 13)]
    for in_len, nan_input in zip(in_lens, nan_inputs):
        inp = torch.randn(1, 1, in_len)
        # inp[0, 0, -1] = torch.nan
        inp[0, 0, slice(*nan_input)] = torch.nan

        out = a(inp)
        solver.add_input_to_output_length(in_len, out.size(2))

        vec = out[0, 0, :].numpy()
        corrupted_idx = np.where(np.isnan(vec))[0]
        left, right = corrupted_idx[0], corrupted_idx[-1] + 1
        print(f"In: {in_len}, Out: {out.size(2)}, Nans: {nan_input} -> {left, right}")
        solver.add_input_to_output_range((nan_input[0], nan_input[1]), (left, right))

    print()
    solver.get_sol()


test_conv1d()
