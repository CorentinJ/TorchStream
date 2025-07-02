import logging

import numpy as np
from torch.nn import ConvTranspose1d

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.nan_trick import get_nan_map
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams, temp_eq
from torchstream.sliding_window.sliding_window_params_solver import (
    SlidingWindowParamsSolver,
)
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream, get_streaming_params
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

trsfm = ConvTranspose1d(1, 1, kernel_size=3, padding=1)
# trsfm = Conv1d(1, 1, kernel_size=2)


# def t(x):
#     return trsfm(torch.nn.functional.pad(x, (2, 0)))

# TODO: solve context being excessive!
real_sol = SlidingWindowParams(kernel_size_out=3, out_trim=1)
# real_sol = SlidingWindowParams(kernel_size_in=2)
print(get_streaming_params(real_sol))


if False or True:
    solver = SlidingWindowParamsSolver(trsfm, SeqSpec((1, 1, -1)), max_hypotheses_per_step=10)
    while solver.nan_trick_params is not None:
        solver.step()
        if len(solver.nan_trick_history) > 5:
            break

    sols = [hypothesis.params for hypothesis in solver.hypotheses]
    for sol in sols:
        print("\n\nSolution:\n", sol)
        print(get_streaming_params(sol))
        print(sol.kernel_in_sparsity)
        print(sol.kernel_out_sparsity)

    if not any(temp_eq(sol, real_sol) for sol in sols):
        hyp = next((hyp for hyp in solver.rejected_hypotheses if temp_eq(hyp.params, real_sol)), None)
        print("==== REAL SOLUTION WAS REJECTED ====" if hyp else "==== REAL SOLUTION WAS NOT FOUND ====")
        print(hyp or real_sol)
        print(get_streaming_params(real_sol))
        for violation in solver.sampler.get_violations(real_sol):
            print(violation)
            print("----")
        print("====")

    quit()


if False:  # or True:
    # in_len, in_nan_idx = find_nan_trick_params_by_infogain(hypotheses)
    # print(f"{in_len=}, {in_nan_idx=}")
    # print()

    # gt_hyp = hypotheses[0]
    # in_nan_idx = 2
    # nan_map = get_nan_map(gt_hyp, 7, (in_nan_idx, in_nan_idx + 1))
    # out_nan_idx = np.where(nan_map > 0)[0]

    for i in hypotheses:
        print("NaN map:", get_nan_map(i, 7, (2, 3)))
        for j, x in enumerate(i.get_inverse_kernel_map(10)):
            print("Inv map", j, x)
        check_nan_trick(i, 7, 3, (2, 3), np.array([0, 1]))
        print("---\n")

    quit()

in_spec = SeqSpec((1, 1, -1))
for _ in range(10):
    print(_)
    test_stream_equivalent(
        trsfm,
        SlidingWindowStream(trsfm, real_sol, in_spec),
        in_step_sizes=tuple(np.random.randint(1, 30, size=200)),
    )
