import inspect
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


st.subheader("1. TorchStream Introduction using Mel-Spectrograms")

"""
A very common speech processing data transformation is the Mel-Spectrogram. ML models that generate or ingest speech 
will typically represent it as a Mel-Spectrogram.
"""
st.space(1)
"""
Let's take a short speech sample from the web:
"""
MP3_URL = "https://d38nvwmjovqyq6.cloudfront.net/va90web25003/companions/ws_smith/32%20Speaking%20The%20Text%20As%20A%20Dramatic%20Reading.mp3"
local_audio_path = download_file_cached(MP3_URL)
wave, sample_rate = load_audio(local_audio_path)
st.audio(wave, sample_rate=sample_rate)
st.caption("Source: https://global.oup.com/us/companion.websites/9780195300505/audio/audio_samples/, sample 32")


def plot_audio(ax, wave: np.ndarray, sample_rate: int):
    time_axis = np.arange(wave.shape[0]) / sample_rate
    ax.plot(time_axis, wave)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(-1, 1)


fig, ax = plt.subplots(figsize=(10, 2.5))
plot_audio(ax, wave, sample_rate)
st.pyplot(fig)

"""
Torchaudio provides a function to compute Mel-Spectrograms from an audio. Let's wrap it in a function with a 
couple useful parameters
"""


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
    )(torch.from_numpy(wave))


st.code(inspect.getsource(get_spectrogram))


"""
The output of that function is a (n_mels, n_frames) shaped tensor that we can view as a 2D image:
"""


def plot_melspec(ax, spec, title="MelSpectrogram"):
    ax.imshow(spec.log2().numpy(), aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bin")
    ax.set_xlim(0, spec.shape[1])
    ax.set_ylim(0, spec.shape[0])


fig, ax = plt.subplots(figsize=(10, 2.5))
plot_melspec(ax, get_spectrogram(wave, sample_rate))
st.pyplot(fig)

"""
And if you're a speech or DSP expert, you could tell from this image alone that this is indeed human speech, from a 
single speaker, with normal prosody, recorded with little noise in the room at a high sample rate.
"""


quit()

"""
An audio file 
"""
# TODO! sharex

st.code(f"{wave.shape} shaped {wave.dtype} numpy array\n{wave[sample_rate : sample_rate + 5]}...")
fig, axs = plt.subplots(figsize=(10, 5.5), nrows=2)
fig.subplots_adjust(hspace=0.5)
axs[0].set_title("Audio")
plot_audio(axs[0], wave, sample_rate)
plot_melspec(axs[1], get_spectrogram(wave, sample_rate))
st.pyplot(fig)

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
