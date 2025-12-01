import logging

import streamlit as st
import torch

from examples.resources.bigvgan.bigvgan import load_uninit_bigvgan
from examples.utils.streamlit_worker import await_running_thread
from torchstream.sequence.sequence import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.subheader("3. Streaming BigVGAN")

"""
In the example #1, we mentioned that mel spectrograms are a common way to represent audio data for neural networks.
Mel spectrograms are cheap to compute, but hard to invert back to audio. This is called vocoding, and modern 
vocoders are neural networks. 

BigVGAN is a state of the art neural vocoder, part of many modern speech synthesis pipelines. For a text-to-speech 
system, streaming it is essential to reduce latency. For a speech-to-speech system, streaming it allows for 
live voice conversion.
"""


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
    left_out_trim=81,
    right_out_trim=81,
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


await_running_thread()
