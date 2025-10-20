import logging

import torch

from examples.bigvgan.bigvgan import load_uninit_bigvgan
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@torch.inference_mode()
def main():
    # Load our model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bigvgan = load_uninit_bigvgan("config_base_22khz_80band", device)

    # Specify the input and output data format
    #   - We'll use a batch size of 1 for finding the parameters
    #   - BigVGAN takes mel-spectrograms as input, with a variable time dimension, so (B, M, T) where M is num_mels
    #   - The output is an audio waveform directly, that's (B, 1, T)
    in_spec = SeqSpec(1, bigvgan.h.num_mels, -1, device=device)
    out_spec = SeqSpec(1, 1, -1, device=device)

    params = SlidingWindowParams(
        kernel_size_in=35,
        stride_in=1,
        left_pad=17,
        right_pad=17,
        kernel_size_out=418,
        stride_out=256,
        out_trim=81,
    )

    # test_stream_equivalent(
    #     bigvgan,
    #     SlidingWindowStream(bigvgan, params, in_spec, out_spec),
    #     in_step_sizes=(7, 4, 12) + (1,) * 100 + (17, 9),
    #     throughput_check_max_delay=params.out_trim,
    # )
    # quit()

    params = find_sliding_window_params(bigvgan, in_spec, out_spec)
    print(params)


main()
