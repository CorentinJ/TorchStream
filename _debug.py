import logging

import numpy as np
from torch.nn import Conv1d, ConvTranspose1d

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.nan_trick import get_nan_map
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import (
    SlidingWindowParamsSolver,
)
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream, get_streaming_params
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

trsfm = ConvTranspose1d(1, 1, kernel_size=3)
trsfm = Conv1d(1, 1, kernel_size=3, stride=3)
real_sol = SlidingWindowParams(kernel_size_in=3, stride_in=3)

# def trsfm(x):
#     x = torch.nn.functional.pad(x, (2, 0))
#     x = conv(x)
#     return x

print(get_streaming_params(real_sol))


if False or True:
    solver = SlidingWindowParamsSolver(trsfm, SeqSpec((1, 1, -1)), max_hypotheses_per_step=10)
    while solver.nan_trick_params is not None:
        solver.step()
        # if len(solver.nan_trick_history) > 3:
        #     break

    sols = [hypothesis.params for hypothesis in solver.hypotheses]
    for sol in sols:
        print("\n\nSolution:\n", sol)
        print(get_streaming_params(sol))
        print(sol.kernel_in_sparsity)
        print(sol.kernel_out_sparsity)
    quit()

    print("==== REAL SOLUTION ====")
    print(real_sol)
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
in_seq = in_spec.new_randn(100)
test_stream_equivalent(
    trsfm,
    SlidingWindowStream(trsfm, SlidingWindowParams(kernel_size_out=3), in_spec),
)
