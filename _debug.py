import logging

import numpy as np
from torch.nn import Conv1d

from tests.rng import set_seed
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.nan_trick import get_nan_map
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import (
    find_sliding_window_params_for_transform,
)
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

set_seed(10)


# trsfm = ConvTranspose1d(1, 1, kernel_size=3, padding=1)
trsfm = Conv1d(1, 1, kernel_size=10)
# conv = trsfm


# def trsfm(x):
#     print("\x1b[31m", x, "\x1b[39m", sep="")
#     return conv(torch.nn.functional.pad(x, (2, 0)))


real_sol = SlidingWindowParams(
    kernel_size_in=10,
    stride_in=1,
    left_pad=0,
    right_pad=0,
    kernel_size_out=1,
    stride_out=1,
    out_trim=0,
)

if False or True:
    sols = find_sliding_window_params_for_transform(trsfm, SeqSpec((1, 1, -1)), debug_ref_params=real_sol)
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
for _ in range(1):
    print(_)
    test_stream_equivalent(
        trsfm,
        SlidingWindowStream(trsfm, real_sol, in_spec),
        # in_step_sizes=tuple(np.random.randint(1, 30, size=200)),
    )
