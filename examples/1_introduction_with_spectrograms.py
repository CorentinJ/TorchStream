import inspect

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchaudio

from examples.utils.animated_sliding_window_stream import AnimatedSlidingWindowStream
from examples.utils.audio import load_audio
from examples.utils.download import download_file_cached
from torchstream import SeqSpec, Sequence, SlidingWindowParams
from torchstream.sliding_window.sliding_window_params import get_streaming_context_size, in_out_size_rel_repr
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

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


def plot_audio(ax, wave: np.ndarray, tick_label_offset_s: float = 0.0):
    ax.plot(wave)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(wave) - 1)
    ax.set_ylim(-1, 1)

    total_seconds = len(wave) / sample_rate
    if total_seconds >= 10:
        increment = 2.0
    elif total_seconds >= 5:
        increment = 1.0
    elif total_seconds >= 1:
        increment = 0.2
    else:
        increment = 0.1
    ticks_seconds = np.arange(0, total_seconds, increment)
    ticks_seconds = np.append(ticks_seconds, round(total_seconds, 1))
    ticks = (ticks_seconds * sample_rate).round().astype(int)
    ax.set_xticks(ticks)
    if total_seconds >= 5:
        ax.set_xticklabels([f"{ts + tick_label_offset_s:.0f}" for ts in ticks_seconds])
    else:
        ax.set_xticklabels([f"{ts + tick_label_offset_s:.1f}" for ts in ticks_seconds])


fig, ax = plt.subplots(figsize=(10, 2.5))
plot_audio(ax, wave)
st.pyplot(fig)

"""
Torchaudio provides a function to compute Mel-Spectrograms from an audio. Let's wrap it in a function with a 
couple useful parameters
"""


def get_spec_trsfm_and_sli_params(
    wave: np.ndarray,
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


st.code(inspect.getsource(get_spec_trsfm_and_sli_params))


"""
The output of that function is a (n_mels, n_frames) shaped tensor that we can view as a 2D image:
"""
melspectrogram = get_spec_trsfm_and_sli_params(wave)
st.code(
    f">>> melspectrogram = get_spectrogram(wave)\n{tuple(melspectrogram.shape)} shaped {melspectrogram.dtype} tensor"
)


def plot_melspec(ax, spec, aspect="auto"):
    spec_np = spec.log2().numpy()
    height, width = spec_np.shape
    ax.imshow(
        spec_np,
        aspect=aspect,
        origin="lower",
        extent=(0, width, 0, height),
        vmin=-20.0,
        vmax=15.0,
    )
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bin")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)


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
C/C++/CUDA** code. Furthermore, the deep neural networks that use such functions arbitrarily combine **several 
hundreds** of them in **deeply nested** python code.

For any complex sequence-to-sequence transform, let alone a fully fledged deep neural network, trying to pick apart 
its inner workings or to change their batch mode of operation (=all input is processed in one go) to streaming is a 
tedious feat of engineering.
"""

"""
Take the docs of our Mel-Spectrogram function. They do not document how the length of the spectrogram is computed:
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
            spec_chunks.append(get_spec_trsfm_and_sli_params(wave[i : i + 1500]))
        except RuntimeError:
            pass  # the last chunk may be too small
    naive_spec = torch.cat(spec_chunks, dim=1)


fig, axs = plt.subplots(figsize=(10, 5.5), nrows=2, sharex=True)
fig.subplots_adjust(hspace=0.5)
axs[0].set_title("Naively Streamed Mel-Spectrogram")
plot_melspec(axs[0], naive_spec)
axs[1].set_title("Original Mel-Spectrogram")
original_spec = get_spec_trsfm_and_sli_params(wave)
plot_melspec(axs[1], original_spec)
axs[0].set_xlim(0, max(naive_spec.shape[1], original_spec.shape[1]))
st.pyplot(fig)
"""
The outputs are not of the same size and there are frequency artifacts at the top and bottom of the naive spectrogram. 
It's a mess.

You might argue that by choosing an appropriate chunk size parameter and adding some overlap between chunks, you could 
obtain the correct output. _You'd be entirely right._

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
at a given position in the output vector. You then offset (_slide_) the input window by a fixed amount (_the stride_) 
to generate the next output. Repeat until the full input has been consumed.
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
"""
When you're working with a transform based on a single (non-trivial) sliding window algorithm, you'll generally be 
able to infer its sliding window parameters by looking at its code or documentation. Let's go with that manual 
approach this once. We'll put TorchStream's solver to use in the next example.

Here are the relevant parts of the torchaudio's MelSpectrogram documentation:
"""
st.code(
    """
class MelSpectrogram(torch.nn.Module):
(...)
Args:
    n_fft (int, optional): Size of FFT, creates n_fft // 2 + 1 bins. (Default: 400)
    win_length (int or None, optional): Window size. (Default: n_fft)
    hop_length (int or None, optional): Length of hop between STFT windows. (Default: win_length // 2)
    center (bool, optional): whether to pad `waveform` on both sides so that frames are centered. (Default: True)
(...)
"""
)

"""
This is rather straightforward. We can retrieve the function and derive its sliding window parameters:
"""

with st.echo():

    def get_spec_trsfm_and_sli_params(n_fft=2048, center: bool = False):
        trsfm = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            center=center,
            n_mels=120,
        )

        sli_params = SlidingWindowParams(
            kernel_size_in=n_fft,
            stride_in=n_fft // 2,
            left_pad=n_fft // 2 if center else 0,
            right_pad=n_fft // 2 if center else 0,
            # Not documented but empirically verifiable: the input size below which the transform will raise an error
            # for not having enough data
            min_input_size=n_fft // 2 + 1 if center else n_fft,
        )

        return trsfm, sli_params


"""
TorchStream can empirically verify whether these parameters give the correct input to output size relationship and 
whether they produce the correct output when streaming. It doesn't _guarantee_ the parameters are true to the transform
nor that they are optimal for streaming, but it's a good smoke test.
"""
with st.echo():
    trsfm, sli_params = get_spec_trsfm_and_sli_params()

    stream = SlidingWindowStream(
        trsfm,
        sli_params,
        # Audio inputs are 1D float arrays of varying length
        input_spec=SeqSpec(-1),
        # Spectrogram outputs are 2D: 120 channels for the mel bins, and varying length for the time frames
        output_spec=SeqSpec(120, -1),
    )

    # This will raise an error if the streamed function differs
    test_stream_equivalent(
        sync_fn=trsfm,
        stream=stream,
        in_data=torch.from_numpy(wave),
        # Take random input chunk sizes (inputs that are too small are not problematic,
        # the stream waits until it has enough to produce an output)
        in_step_sizes=torch.randint(1, 10_000, (20,)).tolist(),
        atol=5e-3,
    )

st.success("Passed the equivalence test")

"""
So there we have it. TorchStream took a function not originally designed for streaming and it inferred how it should 
be streamed based on its sliding window parameters. Let's see what that looks like under the hood.
"""

with st.container(border=True):
    total_seconds = len(wave) / sample_rate
    min_slice_seconds = 0.1
    start_sec, end_sec = st.slider(
        "Select the audio segment",
        0.0,
        total_seconds,
        (0.25, 0.95),
        step=0.1,
        format="%.1fs",
    )
    if end_sec - start_sec < min_slice_seconds:
        end_sec = min(total_seconds, start_sec + min_slice_seconds)
        if end_sec - start_sec < min_slice_seconds:
            start_sec = max(0.0, end_sec - min_slice_seconds)
    start_sample = int(start_sec * sample_rate)
    end_sample = max(start_sample + int(min_slice_seconds * sample_rate), int(end_sec * sample_rate))

    wave_slice = torch.from_numpy(wave[start_sample:end_sample])

    with st.container(border=True):
        """
        Set the transform's parameters
        """
        left_col, right_col = st.columns([0.25, 0.75])
        with left_col:
            n_fft = st.slider("n_fft", 64, 4096, value=sli_params.kernel_size_in)
            center = st.checkbox("center", value=True)

        trsfm, sli_params = get_spec_trsfm_and_sli_params(n_fft=n_fft, center=center)
        with right_col:
            st.code(
                str(sli_params)
                + f"\n-> min/max overlap: {[sli_params.streaming_context_size, sli_params.streaming_context_size + sli_params.stride_in - 1]}\n"
                + f"-> min/max output delay: {list(sli_params.output_delay_bounds)}\n"
                + f"-> in/out size relation: {in_out_size_rel_repr(*sli_params.canonical_in_out_size_params)}"
            )

    def build_stream(chunk_size: int) -> AnimatedSlidingWindowStream:
        stream_obj = AnimatedSlidingWindowStream(
            trsfm,
            sli_params,
            SeqSpec(-1),
            SeqSpec(120, -1),
        )
        stream_obj.forward_in_chunks(Sequence(wave_slice, seq_dim=0), chunk_size=chunk_size)
        return stream_obj

    chunk_size = st.slider(
        "Set the streaming chunk size",
        sli_params.min_input_size,
        len(wave_slice) // 2,
        value=sli_params.kernel_size_in * 2,
    )
    stream = build_stream(chunk_size)

    @st.fragment
    def stream_step_fragment():
        fig, axs = plt.subplots(figsize=(10, 8.5), nrows=3)
        fig.subplots_adjust(hspace=0.5)

        plot_placeholder = st.empty()

        step_idx = (
            st.slider(
                "Streaming step",
                1,
                len(stream.step_history),
                value=min(2, len(stream.step_history) - 1),
            )
            - 1
        )
        fig.suptitle(
            f"Streaming torchaudio.MelSpectrogram - Step {step_idx + 1}/{len(stream.step_history)}", fontsize=18
        )

        # Input plot
        plot_audio(axs[0], wave_slice.numpy(), tick_label_offset_s=start_sec)

        # Sync spectrogram plot
        plot_melspec(axs[2], trsfm(wave_slice))

        stream.plot_step(step_idx, *axs, out_plot_fn=plot_melspec)
        plot_placeholder.pyplot(fig)

    stream_step_fragment()

"""
#### Streaming context size

In the above demo you'll see that our input chunks overlap eachother quite a bit. The blue arrow displays the 
rightmost section of the _previous_ input chunk that is reused in the current input. Hence overlap means: "how much 
of the previous input is kept in a **buffer** inbetween steps". TorchStream automatically manages this buffer 
inside the stream object and ensures its size remains **minimal**.

The minimum size of this buffer is called the **streaming context size**. This amount of context is **intrinsically 
needed by the stream** to produce the same output as the non-streaming function. In general, the streaming context 
size is non trivial to determine.
"""

with st.expander("See the source code of `get_streaming_context_size()`"):
    st.code(inspect.getsource(get_streaming_context_size))

with st.container(border=True):
    """
    A suboptimal streaming implementation might use an excessive amount of context, leading to a higher computational 
    **overhead** that is particularly impactful on **smaller chunk sizes**.
    """

"""
#### Output delay

You might have noticed that the "output to discard" rectangles appear on each step only when the center parameter is 
set to True. This is because enabling `center` adds half a window of padding on each side of every intermediary 
input, which does not happen during the non-streaming function call, leading to skewed outputs.

The output delay is the amount of output that we need to discard on the right of intermediary outputs, and therefore 
it also acts as a measure of overhead.

"""

with st.container(border=True):
    """
    Input padding and output trimming are sources of **overhead** that can be **significant** just like they can be 
    **negligible**. 
    
    When they are significant, one should consider applying padding and trimming _outside_ of the streamed transform.
    """


"""
### Up next
You've seen how we can stream a simple transform with known sliding window parameters. The following examples will 
cover **automatic derivation of sliding window parameters** and handling of **complex transforms**, including those that 
are **not sliding window based**.
"""
