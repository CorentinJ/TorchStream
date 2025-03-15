import torch
from colorama import Fore
from torch import nn

from torchstream.sliding_window.nan_trick import check_nan_trick, get_nan_range
from torchstream.sliding_window.sliding_window_params_solver import SlidingWindowParamsSolver


@torch.no_grad()
def test_conv1d():
    a = nn.Conv1d(1, 1, kernel_size=5, stride=2)
    solver = SlidingWindowParamsSolver()

    # TODO: edge cases
    in_lens = (80, 120, 120, 120, 200, 2)  # 14, 17)
    nan_inputs = [(0, 1), (8, 50), (9, 51), (10, 48), (199, 200), (1, 2)]  # (5, 10), (11, 13)]
    out_lens = []
    out_ranges = []
    for in_len, nan_input in zip(in_lens, nan_inputs):
        inp = torch.randn(1, 1, in_len)
        inp[0, 0, slice(*nan_input)] = torch.nan

        out = a(inp)
        out_lens.append(out.size(2))

        left, right = get_nan_range(out)
        if left is None:
            raise ValueError("No NaNs in output")
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
            success, reason = check_nan_trick(params, in_len, out_len, nan_input, out_range)
            if not success:
                print(f"{Fore.RED}Failed!{Fore.RESET} Reason: {reason}")
                failed = True

        if not failed:
            print(f"{Fore.GREEN}Success!{Fore.RESET}")
            n_sols += 1

    print(f"\nFound {n_sols}/{len(all_params)} working solutions")


test_conv1d()
test_conv1d()
