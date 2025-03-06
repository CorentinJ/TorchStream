from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from z3 import Int, Ints, Or, Solver, sat

# TODO: support multiple inputs/output as long as they have the same sequence length


def printsol(model):
    var_dict = {str(decl): model[decl] for decl in model.decls()}
    print("--- Solution ---")
    var_right_pad_names = [var for var in var_dict if var.startswith("p_r_")]
    # FIXME: alphanum
    var_right_pad_vals = [str(var_dict[var]) for var in sorted(var_right_pad_names)]
    print(f"Padding: left={var_dict['p_l']}, right={var_dict['p_r']} (var={', '.join(var_right_pad_vals)})")
    print(f"Input: kernel_size={var_dict['k_i']}, stride={var_dict['s_i']}")
    print(f"Output: kernel_size={var_dict['k_o']}, stride={var_dict['s_o']}")
    print()


def determine_sliding_window_params_from_nan_trick(
    input_lens: Iterable[int],
    output_lens: Iterable[int],
    nan_input_ranges: Iterable[Tuple[int, int]],
    nan_output_ranges: Iterable[Tuple[int, int]],
):
    input_lens = list(input_lens)
    output_lens = list(output_lens)
    nan_input_ranges = [tuple(map(int, nan_input_range)) for nan_input_range in nan_input_ranges]
    nan_output_ranges = [tuple(map(int, nan_output_range)) for nan_output_range in nan_output_ranges]
    if not (len(input_lens) == len(output_lens) == len(nan_input_ranges) == len(nan_output_ranges)):
        raise ValueError("Arguments must all be of the same length")

    # TODO: more check on values: range, ints, etc...

    # Define the parameters we're trying to uniquely determine
    solver = Solver()
    # k_i and s_i are respectively the input kernel size and stride (NOTE: it's technically the kernel span, i.e. the
    # whole span of the kernel if dilation > 1)
    k_i, s_i = Ints("k_i s_i")
    # It would be highly unusual to have a stride larger than the kernel size, leading to inputs being dropped.
    solver.add(k_i >= s_i, s_i > 0)
    # k_o and s_o are respectively the output kernel size and stride. These are both 1 for normal convolutions,
    # but not for transposed convolutions.
    k_o, s_o = Ints("k_o s_o")
    # Again, it would be strange to have a stride larger than the kernel size, leading to gaps in the output.
    solver.add(k_o >= s_o, s_o > 0)
    # The left and right input padding. Note that the right padding can vary in practice (we account for that below).
    # I have not yet seen a case where varying the left padding is useful, so we'll assume it constant.
    # There is no point in making the padding higher than the kernel size, as it would waste compute on constant values.
    p_l, p_r = Ints("p_l p_r")
    solver.add(p_l >= 0, p_r >= 0, p_l < k_i, p_r < k_i)

    for input_idx, (in_len, out_len, nan_in_range, nan_out_range) in enumerate(
        zip(input_lens, output_lens, nan_input_ranges, nan_output_ranges)
    ):
        # Account for variable right padding
        p_r_var = Int(f"p_r_{input_idx}")
        solver.add(p_r_var >= 0, p_r_var < p_r)

        ## Input length -> output length deductions
        solver.add(in_len + p_l + p_r_var >= k_i)
        solver.add(out_len == ((in_len + p_l + p_r_var - k_i) / s_i) * s_o + k_o)

        ## NaN trick
        # The number of windows that overlap with the nan region
        c_nan = Int(f"c_nan_{input_idx}")

        # The number of windows that overlap with the nan region either be the ceil or floor of the division
        # below (z3 uses floor division on ints)
        nan_in_len = nan_in_range[1] - nan_in_range[0]
        solver.add(
            Or(
                c_nan == (nan_in_len + k_i - 1) / s_i,
                c_nan == ((nan_in_len + k_i - 1) + s_i - 1) / s_i,
            )
        )
        # From the number of windows, we uniquely determine the nan output size
        nan_out_len = nan_out_range[1] - nan_out_range[0]
        solver.add(nan_out_len == (c_nan - 1) * s_o + k_o)

        # We can also make deductions based on the nan start position in the input vs. the output
        if nan_out_range[0] == 0:
            solver.add(nan_in_range[0] + p_l <= k_i)
        else:
            solver.add(nan_out_range[0] == ((nan_in_range[0] - k_i + 1) + s_i - 1) / s_i)

        # TODO: constraint on nan start/end pos relative to padding

    if solver.check() == sat:
        model1 = solver.model()
        printsol(model1)
        solver.add(
            Or(
                k_i != model1[k_i],
                k_o != model1[k_o],
                s_i != model1[s_i],
                s_o != model1[s_o],
            )
        )
        if solver.check() == sat:
            printsol(solver.model())
        else:
            print("The solution is unique.")
    else:
        print("No solution")


@torch.no_grad()
def test_conv1d():
    a = nn.Conv1d(1, 1, kernel_size=7, stride=4)

    in_lens = (50, 80, 101)
    nan_inputs = [(25, 26), (30, 35), (60, 62)]
    out_lens = []
    nan_outputs = []
    for in_len, nan_input in zip(in_lens, nan_inputs):
        inp = torch.randn(1, 1, in_len)
        inp[0, 0, slice(*nan_input)] = torch.nan

        out = a(inp)
        out_lens.append(out.size(2))

        vec = out[0, 0, :].numpy()
        corrupted_idx = np.where(np.isnan(vec))[0]
        left, right = corrupted_idx[0], corrupted_idx[-1] + 1
        print(f"Nans at {nan_input} -> {left, right}")
        nan_outputs.append((left, right))

    print()
    determine_sliding_window_params_from_nan_trick(in_lens, out_lens, nan_inputs, nan_outputs)


test_conv1d()
