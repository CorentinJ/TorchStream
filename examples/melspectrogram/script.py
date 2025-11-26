import inspect
import logging

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchaudio
from matplotlib.patches import ConnectionPatch, Rectangle

from demo_tools.audio import load_audio
from demo_tools.download import download_file_cached
from demo_tools.sliding_window_stream_with_history import SlidingWindowStreamWithHistory
from torchstream import SeqSpec, Sequence, SlidingWindowParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.subheader("1. Introduction to TorchStream")

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
melspectrogram = get_spectrogram(wave, sample_rate)
st.code(
    f">>> melspectrogram = get_spectrogram(wave, sample_rate)\n"
    f"{tuple(melspectrogram.shape)} shaped {melspectrogram.dtype} tensor"
)


def plot_melspec(ax, spec, aspect="auto"):
    ax.imshow(spec.log2().numpy(), aspect=aspect, origin="lower")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bin")
    ax.set_xlim(0, spec.shape[1])
    ax.set_ylim(0, spec.shape[0])


fig, ax = plt.subplots(figsize=(10, 2.5))
plot_melspec(ax, melspectrogram)
st.pyplot(fig)

"""
If you're a speech or DSP expert, you could tell from this image alone that this is indeed human speech, from a 
single speaker, with normal prosody, recorded with little room noise at a high sample rate.

But this is not our concern here.
"""

st.markdown("### What TorchStream is for")

"""
If you're working on an application involving a sequence-to-sequence transform such as this one, you might want to
- **Explain which inputs produced which outputs** for analysis, debugging or interpretability of results
- **Derive parameters of the transform** such as its input to output size relationship or its receptive field
- **Stream the computation** on incoming live input, or on static input for low latency

And you'd be stuck for a while. The torch and numpy functions that make the backbone of modern Machine Learning 
and AI are **vectorized**, wrapped in **countless layers of abstractions** and implemented in **highly optimized 
C/C++/CUDA** code. Trying to pick apart their inner workings or to change their batch mode of operation (=all input 
is processed in one go) to streaming is a tedious feat of engineering.
"""

"""
For instance, the torchaudio docs do not document the output size of their Spectrogram transform:
> Returns Dimension (…, freq, time), where freq is n_fft // 2 + 1 where n_fft is the number of Fourier bins, and time 
is the number of window hops (n_frame).

Nor does [librosa](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html#librosa-feature-melspectrogram), 
which also implements Mel-Spectrograms:
> S: np.ndarray [shape=(…, n_mels, t)]

And if you try to stream our function by naively forwarding chunks of your input data, you'll be disappointed with the 
results:
"""

with st.echo():
    spec_chunks = []
    for i in range(0, wave.shape[0], 1500):
        try:
            spec_chunks.append(get_spectrogram(wave[i : i + 1500], sample_rate))
        except RuntimeError:
            pass  # the last chunk may be too small
    naive_spec = torch.cat(spec_chunks, dim=1)


fig, axs = plt.subplots(figsize=(10, 5.5), nrows=2, sharex=True)
fig.subplots_adjust(hspace=0.5)
axs[0].set_title("Naively Streamed Mel-Spectrogram")
plot_melspec(axs[0], naive_spec)
axs[1].set_title("Original Mel-Spectrogram")
original_spec = get_spectrogram(wave, sample_rate)
plot_melspec(axs[1], original_spec)
axs[0].set_xlim(0, max(naive_spec.shape[1], original_spec.shape[1]))
st.pyplot(fig)
"""
The outputs are not of the same size and there are frequency artifacts at the top and bottom of the naive spectrogram. 
It's a mess.

You might argue that by choosing an appropriate chunk size parameter and adding some overlap between chunks, you could 
recover the correct output. _You'd be entirely right._

But what are these parameters? How do we obtain them? How can we be sure they are correct and optimal? And how can 
we stream not a mere spectrogram function but **real-world massive neural networks with hundreds of layers of data 
transformation**?
"""
st.html('<div style="text-align: center">This is what TorchStream is for.</div>')

st.markdown("### It's (almost) all sliding windows")

"""
The Mel-Spectrogram transform is an example of a **sliding window algorithm**. 
"""
st.image("examples/_resources/sli_algo1.png")

"""
You take a slice of fixed size of the input data (_a window_), apply a function (_a kernel_) on it and store the output 
at a given position in the output vector. You then offset (_slide_) the input window and the output position by 
fixed amounts (_the stride_) and repeat.
"""
st.image("examples/_resources/sli_algo2.png")

"""
This is a textbook definition of a sliding window algorithm. **It's a limiting one**; machine learning engineers deal 
with sliding window algorithms on a daily basis and they often don't realise it. Let's augment our model with
- An output kernel size and stride that can be larger than 1
- Padding on the left and right of the full input
- Trimming on the left and right of the full output
- Kernels that can skip over input elements
"""
st.image("examples/_resources/sli_algo3.png")

"""
And **almost everything becomes a sliding window**:
- A dilated (à trous) convolution? That's a sliding window with a sparse kernel.
- A transposed convolution? That's a sliding window with some output kernels, strides and trimming.
- A pooling or upsampling layer? That's a sliding window with an input or output stride that's greater than 1.
- Adding boundary tokens to a sequence? That's just a form of padding.
- An element-wise operation? A special case of sliding window with kernel size 1.

Any transform that falls under this model is **streamable**, and has a well-defined **input to output mapping**.
"""

with st.container(border=True, vertical_alignment="center", gap="medium"):
    """
    A core part of TorchStream is the `find_sliding_window_params` function. It will **automatically find for you** the 
    sliding window parameters of the parts of your pipeline that behave as described above. For any remaining 
    non sliding window transform, TorchStream aims to give you the **tools necessary** to implement their **streaming 
    version**, an **approximation** of it, or if that is to be the case, to understand quickly why **streaming them 
    is not possible**.
    """

st.markdown("### Back to our example")

wave = wave[sample_rate // 5 : sample_rate]


trsfm = lambda x: get_spectrogram(x, sample_rate, n_fft=2048)

params = SlidingWindowParams(
    kernel_size_in=2048,
    stride_in=1024,
    left_pad=1024,
    right_pad=1024,
)


n_steps = 5
stream = SlidingWindowStreamWithHistory(
    trsfm,
    params,
    SeqSpec(-1, dtype=np.float32),
    SeqSpec(120, -1),
)
stream.forward_in_chunks(Sequence(wave, seq_dim=0), chunk_size=len(wave) // n_steps)

fig, axs = plt.subplots(figsize=(10, 8.5), nrows=3)
fig.subplots_adjust(hspace=0.5)

# Input plot
plot_audio(axs[0], wave, sample_rate)

# Sync spectrogram plot
plot_melspec(axs[2], trsfm(wave))


# Streamed plot
def plot_stream_step(
    in_ax,
    out_stream_ax,
    out_sync_ax,
    in_buff_start_pos: int,
    in_new_start_pos: int,
    in_buff_drop_pos: int,
    in_end_pos: int,
    out_start_pos: int,
    out_size: int,
    out_trim_start: int,
    out_trim_end: int,
    untrimmed_output: Sequence,
):
    in_ax.axvline(in_buff_start_pos / sample_rate, color="purple", linestyle="--")
    in_ax.axvline(in_new_start_pos / sample_rate, color="blue", linestyle="--")
    in_ax.axvline(in_end_pos / sample_rate, color="purple", linestyle="--")

    out_stream_ax.imshow(untrimmed_output.data[0].log2().numpy(), aspect="auto", origin="lower")

    # Align the middle plot with the input slice
    to_fig = fig.transFigure.inverted()
    start_fig_x = to_fig.transform(in_ax.transData.transform((in_buff_start_pos / sample_rate, 0)))[0]
    end_fig_x = to_fig.transform(in_ax.transData.transform((in_end_pos / sample_rate, 0)))[0]
    orig_pos = out_stream_ax.get_position()
    out_stream_ax.set_position([start_fig_x, orig_pos.y0, end_fig_x - start_fig_x, orig_pos.height])

    # Remove the ticks & border on the middle plot
    out_stream_ax.set_xticks([])
    out_stream_ax.set_yticks([])
    for spine in out_stream_ax.spines.values():
        spine.set_visible(False)

    # Draw the trimming on the output
    if out_size:

        def add_trim_rect(start_frac: float, end_frac: float):
            start = max(0.0, min(1.0, start_frac))
            end = max(0.0, min(1.0, end_frac))
            width = end - start
            if width <= 0:
                return
            rect = Rectangle(
                (start, 0),
                width,
                1,
                transform=out_stream_ax.transAxes,
                edgecolor="red",
                facecolor=(1, 0, 0, 0.12),
                hatch="///",
                linewidth=1.2,
                zorder=4,
            )
            out_stream_ax.add_patch(rect)

        add_trim_rect(0.0, out_trim_start / out_size)
        add_trim_rect(out_trim_end / out_size, 1.0)

    # Connect the highlighted input boundaries to the top corners of the middle image
    input_bottom = in_ax.get_ylim()[0]
    connections = [
        (in_buff_start_pos / sample_rate, (0, 1)),
        (in_end_pos / sample_rate, (1, 1)),
    ]
    for x_value, corner in connections:
        fig.add_artist(
            ConnectionPatch(
                xyA=(x_value, input_bottom),
                xyB=corner,
                coordsA="data",
                coordsB="axes fraction",
                axesA=in_ax,
                axesB=out_stream_ax,
                color="purple",
                linestyle="--",
            )
        )
    output_top = out_sync_ax.get_ylim()[1]
    connections = [
        (out_start_pos, (0, 0)),
        (out_start_pos + out_size, (1, 0)),
    ]
    for x_value, corner in connections:
        fig.add_artist(
            ConnectionPatch(
                xyA=(x_value, output_top),
                xyB=corner,
                coordsA="data",
                coordsB="axes fraction",
                axesA=out_sync_ax,
                axesB=out_stream_ax,
                color="red",
                linestyle="--",
            )
        )


plot_stream_step(*axs, **stream.step_history[1])


st.pyplot(fig)

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
# params = SlidingWindowParams(
#     kernel_size_in=64,
#     stride_in=64,
#     left_pad=0,
#     right_pad=0,
#     kernel_size_out=8,
#     stride_out=1,
#     left_out_trim=7,
#     right_out_trim=7,
# )

# test_stream_equivalent(
#     transform,
#     SlidingWindowStream(transform, params, in_spec, out_spec),
#     in_data=in_spec.new_randn_arrays(params.min_input_size * 2),
#     throughput_check_max_delay=params.right_out_trim,
#     atol=1e-3,
# )
