import logging

import torch

from examples.bigvgan.bigvgan import load_uninit_bigvgan
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import (
    find_sliding_window_params_for_transform,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    device = torch.device("cpu")
    # device = torch.device("cuda")
    bigvgan = load_uninit_bigvgan("config_base_22khz_80band", device)

    in_spec = SeqSpec((1, bigvgan.h.num_mels, -1), device=device)
    # out_spec = SeqSpec((1, 256, -1), device=device)
    out_spec = SeqSpec((1, 1, -1), device=device)

    with torch.no_grad():
        x = in_spec.new_randn(40)
        print(device)
        print(x.shape)
        y_g_hat = bigvgan(x)
        print(y_g_hat.shape)

        params = find_sliding_window_params_for_transform(bigvgan, in_spec, out_spec)
        print(params)


main()
