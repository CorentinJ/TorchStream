from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from z3 import Int, Ints, Or, Solver, sat

# TODO: support multiple inputs/output as long as they have the same sequence length


def printsol(model):
    var_dict = {str(decl): model[decl] for decl in model.decls()}
    print("--- Solution ---")
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

    # TODO: account for literal edge cases
    nan_input_lens = [nan_input_range[1] - nan_input_range[0] for nan_input_range in nan_input_ranges]
    nan_output_lens = [nan_output_range[1] - nan_output_range[0] for nan_output_range in nan_output_ranges]
    print(nan_input_lens, nan_output_lens)

    k_i, k_o, s_i, s_o = Ints("k_i k_o s_i s_o")
    solver = Solver()
    solver.add(k_i >= s_i, k_o >= s_o, s_i > 0, s_o > 0)
    for i, (in_len, out_len) in enumerate(zip(nan_input_lens, nan_output_lens)):
        c = Int(f"c{i}")
        # Constraint on c, the number of windows that overlap with the nan region. It can be either the ceil or floor
        # of the division below (z3 uses floor division on ints)
        solver.add(
            Or(
                c == (in_len + k_i - 1) / s_i,
                c == ((in_len + k_i - 1) + s_i - 1) / s_i,
            )
        )
        # Constraint on the output length, based on c
        solver.add(out_len == (c - 1) * s_o + k_o)

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
    a = nn.Conv1d(1, 1, kernel_size=6, stride=4)

    in_lens = (50, 80, 100)
    nan_idx = (25, 30, 60)
    nan_inputs = [(nan_idx, nan_idx + 1) for nan_idx in nan_idx]
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
