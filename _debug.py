import logging

from torch.nn import Conv1d

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import (
    find_nan_trick_params_by_infogain,
    find_sliding_window_params_for_transform,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# solver = SlidingWindowParamsSolver()
# solver.add_in_out_range_map((8, 12), [])
# hyps = solver.get_solutions()
# sols = find_nan_trick_params_by_infogain(hyps)

a = Conv1d(1, 1, kernel_size=4, stride=4)
sol = find_sliding_window_params_for_transform(a, SeqSpec((1, 1, -1)))  # , max_hypotheses_per_step=10)
print(sol)
quit()


b = (
    SlidingWindowParams(
        kernel_size_in=5,
        stride_in=5,
        left_pad=1,
        right_pad=4,
    ),
    SlidingWindowParams(
        kernel_size_in=7,
        stride_in=5,
        left_pad=3,
        right_pad=4,
    ),
    SlidingWindowParams(
        kernel_size_in=6,
        stride_in=5,
        left_pad=2,
        right_pad=4,
    ),
)

print(find_nan_trick_params_by_infogain(list(b)))
print()

for i in b:
    print(i)
    for j, x in enumerate(i.get_kernel_map(10)):
        print(j, x)
    print("---")
