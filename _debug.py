import logging

from torch.nn import Conv1d

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import (
    find_sliding_window_params_for_transform,
)
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

trsfm = Conv1d(1, 1, kernel_size=2, stride=2, dilation=2)

if False or True:
    sol = find_sliding_window_params_for_transform(trsfm, SeqSpec((1, 1, -1)))  # , max_hypotheses_per_step=10)
    print("\nSolution:\n", sol)
    quit()


hypotheses = [
    SlidingWindowParams(
        kernel_size_in=1,
        left_pad=0,
        right_pad=0,
        kernel_size_out=2,
    ),
    SlidingWindowParams(
        kernel_size_in=2,
        left_pad=1,
        right_pad=1,
        kernel_size_out=1,
    ),
]

if False or True:
    # in_len, in_nan_idx = find_nan_trick_params_by_infogain(hypotheses)
    # print(f"{in_len=}, {in_nan_idx=}")
    # print()

    # gt_hyp = hypotheses[-1]
    # nan_map = get_nan_map(gt_hyp, in_len, (in_nan_idx, in_nan_idx + 1))
    # out_nan_idx = np.where(nan_map > 0)[0]

    for i in hypotheses:
        print(i)
        for j, x in enumerate(i.get_inverse_kernel_map(7)):
            print(j, x)
        # compat = check_nan_trick(i, in_len, len(nan_map), (in_nan_idx, in_nan_idx + 1), out_nan_idx)
        # print(f"  {compat=}")
        print("---")

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
