import logging

import numpy as np
import torch
from torch.nn import Conv1d

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

# trsfm = Conv1d(1, 1, kernel_size=4, stride=2, dilation=2)
# trsfm = Conv1d(1, 1, kernel_size=2)
# trsfm = ConvTranspose1d(1, 1, kernel_size=2)

conv = Conv1d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=2,
    dilation=2,
)


def trsfm(x):
    x = torch.nn.functional.pad(x, (2, 0))
    x = conv(x)
    return x


if False or True:
    solver = SlidingWindowParamsSolver(trsfm, SeqSpec((1, 1, -1)), max_hypotheses_per_step=20)
    while solver.nan_trick_params is not None:
        solver.step()
        # if len(solver.nan_trick_history) > 3:
        #     break

    # sols = [hypothesis.params for hypothesis in solver.hypotheses]
    # assert sols
    # for sol in sols:
    #     print("\n\nSolution:\n", sol)
    #     print(get_streaming_params(sol))
    #     print(sol.kernel_in_sparsity)
    #     print(sol.kernel_out_sparsity)

    sol = SlidingWindowParams(kernel_size_in=5, stride_in=2, left_pad=2)
    print("====")
    print(sol)
    print(get_streaming_params(sol))
    for violation in solver.sampler.get_violations(sol):
        print(violation)
        print("----")
    print("====")

    quit()


if False or True:
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
for hypothesis in list(hypotheses):
    try:
        test_stream_equivalent(
            trsfm,
            SlidingWindowStream(trsfm, hypothesis, in_spec),
            in_seq,
        )
    except AssertionError as e:
        print(f"Failed for hypothesis {hypothesis}: {e}")
