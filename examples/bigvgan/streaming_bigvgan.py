import logging

import torch

from examples.bigvgan.bigvgan import load_uninit_bigvgan
from tests.stream_equivalence import test_stream_equivalent
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import (
    find_sliding_window_params_for_transform,
    group_sli_params_by_inv_map,
)
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream

logger = logging.getLogger(__name__)


def main():
    device = torch.device("cpu")
    bigvgan = load_uninit_bigvgan("config_base_22khz_80band", device)
    # print(bigvgan.conv_pre)
    # print(bigvgan.ups[0][0])
    # quit()

    # in_spec = SeqSpec((1, bigvgan.h.num_mels, -1), device=device)
    in_spec = SeqSpec((1, 80, -1), device=device)
    out_spec = SeqSpec((1, 256, -1), device=device)

    with torch.no_grad():
        x = in_spec.new_randn(40)
        print(device)
        print(x.shape)
        y_g_hat = bigvgan(x)
        print(y_g_hat.shape)

        logging.basicConfig(level=logging.DEBUG)

        params = find_sliding_window_params_for_transform(bigvgan, in_spec, out_spec, max_in_kernel_gap=1)
        print(params)

        print(group_sli_params_by_inv_map(params, 28))
        return

        for param in params:
            try:
                test_stream_equivalent(
                    bigvgan,
                    SlidingWindowStream(bigvgan, param, in_spec, out_spec),
                    # check_throughput_with_nan_trick=True,
                )
                print(f"Passed for {param}")
            except AssertionError as e:
                print(repr(e))
                print("Failed")
                continue


main()
