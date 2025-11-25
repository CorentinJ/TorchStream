import logging

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchaudio

from demo_tools.audio import load_audio
from demo_tools.download import download_file_cached
from torchstream import (
    SeqSpec,
    SlidingWindowParams,
    SlidingWindowStream,
    test_stream_equivalent,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MP3_URL = "https://d38nvwmjovqyq6.cloudfront.net/va90web25003/companions/ws_smith/32%20Speaking%20The%20Text%20As%20A%20Dramatic%20Reading.mp3"


local_audio_path = download_file_cached(MP3_URL)


wave, sample_rate = load_audio(local_audio_path)
st.audio(wave, sample_rate=sample_rate)
wave = torch.from_numpy(wave)


def get_spectrogram(
    wave: np.ndarray,
    sample_rate: int,
    n_mels=120,
    n_fft=2048,
    hop_size: int | None = None,
    center: bool = True,
):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        center=center,
        n_mels=n_mels,
        hop_length=hop_size,
    )(wave)


def plot_melspec(spec, title="MelSpectrogram"):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(spec.log2().numpy(), aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bin")
    st.pyplot(fig)


plot_melspec(get_spectrogram(wave, sample_rate))
quit()


in_spec = SeqSpec(-1)
out_spec = SeqSpec(n_mels, -1)
# solutions = find_sliding_window_params(
#     transform,
#     in_spec,
#     out_spec,
#     zero_size_exception_signatures=[(RuntimeError, "expected 0 < n_fft <")],
# )
# print(solutions)
# params = solutions[0]
params = SlidingWindowParams(
    kernel_size_in=64,
    stride_in=64,
    left_pad=0,
    right_pad=0,
    kernel_size_out=8,
    stride_out=1,
    left_out_trim=7,
    right_out_trim=7,
)

test_stream_equivalent(
    transform,
    SlidingWindowStream(transform, params, in_spec, out_spec),
    in_data=in_spec.new_randn_arrays(params.min_input_size * 2),
    throughput_check_max_delay=params.right_out_trim,
    atol=1e-3,
)
