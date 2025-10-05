import logging

import torch

from examples.bigvgan.bigvgan import load_uninit_bigvgan
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import (
    find_sliding_window_params,
)
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    device = torch.device("cpu")
    # device = torch.device("cuda")
    bigvgan = load_uninit_bigvgan("config_base_22khz_80band", device)

    in_spec = SeqSpec((1, bigvgan.h.num_mels, -1), device=device)
    out_spec = SeqSpec((1, 1, -1), device=device)

    params = SlidingWindowParams(
        kernel_size_in=35,
        stride_in=1,
        left_pad=17,
        right_pad=17,
        kernel_size_out=418,
        stride_out=256,
        out_trim=81,
    )

    test_stream_equivalent(
        bigvgan,
        SlidingWindowStream(bigvgan, params, in_spec, out_spec),
        in_step_sizes=(7, 4, 12) + (1,) * 100 + (17, 9),
        throughput_check_max_delay=params.out_trim,
    )
    quit()

    with torch.no_grad():
        x = in_spec.new_randn(40)
        print(device)
        print(x.shape)
        y_g_hat = bigvgan(x)
        print(y_g_hat.shape)

        params = find_sliding_window_params(bigvgan, in_spec, out_spec)
        print(params)


main()
