import logging

from torch.nn import Conv1d

from tests.rng import set_seed
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import (
    SlidingWindowParams,
)
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params_for_transform

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

set_seed(10)


other = SlidingWindowParams(5, 2)


trsfm = Conv1d(1, 1, kernel_size=5, stride=2)
# conv = trsfm


find_sliding_window_params_for_transform(trsfm, SeqSpec((1, 1, -1)), debug_ref_params=other)

# def trsfm(x):
#     # print("\x1b[31m", x, "\x1b[39m", sep="")
#     return conv(torch.nn.functional.pad(x, (1, 4)))


# print(
#     test_stream_equivalent(
#         trsfm,
#         SlidingWindowStream(
#             trsfm,
#             ref,
#             SeqSpec((1, 1, -1)),
#         ),
#         in_step_sizes=[randint(1, 200) for _ in range(100)],
#     ),
# )

quit()


real_sol = SlidingWindowParams(kernel_size_in=3, left_pad=2)

if False:  # or True:
    sols = find_sliding_window_params_for_transform(trsfm, SeqSpec((1, 1, -1)), debug_ref_params=real_sol)
    quit()


def f(x):
    x = (x + 2) // 3 + 3
    x = (x - 2) * 2 - 3
    x = (x + 2) // 4 + 3
    x = (x - 3) * 7 - 1
    return max(x, 0)


in_out_rel_sampler = SlidingWindowInOutRelSampler()
in_out_rel_sampler.add_in_out_size(1, f(1))

shape_params_hyps = []
for step in range(1, 1000000):
    shape_params_hyps = in_out_rel_sampler.get_new_solutions(shape_params_hyps)
    print("step", step, shape_params_hyps)

    if len(shape_params_hyps) <= 1:
        params = shape_params_hyps[0]

        def g(x):
            return max(0, ((x + params[2]) // params[0]) * params[1] + params[3])

        for i in range(200):
            assert f(i) == g(i), (i, f(i), g(i))
        print("passed")
        quit()

    # TODO: use infogain
    np_si, np_so, np_isbc, np_osbc = [np.array(param_group)[..., None] for param_group in zip(*shape_params_hyps)]
    # FIXME! size
    out_sizes = np.stack([np.arange(1, 1000)] * len(shape_params_hyps))
    out_sizes = np.maximum(((out_sizes + np_isbc) // np_si) * np_so + np_osbc, 0)
    unique_counts = [len(np.unique(out_sizes[:, i])) for i in range(out_sizes.shape[1])]
    in_size = int(np.argmax(unique_counts)) + 1
    assert unique_counts[in_size - 1] > 1

    out_size = f(in_size)
    in_out_rel_sampler.add_in_out_size(in_size, out_size)

    # Exclude known solutions
    prev_n_hyps = len(shape_params_hyps)
    shape_params_hyps = [
        params for idx, params in enumerate(shape_params_hyps) if out_sizes[idx, in_size - 1] == out_size
    ]
    assert prev_n_hyps > len(shape_params_hyps), "Internal error: did not reject any shape hypotheses"
    print(f"Step {step}: rejected {prev_n_hyps - len(shape_params_hyps)}/{prev_n_hyps} hypotheses")
