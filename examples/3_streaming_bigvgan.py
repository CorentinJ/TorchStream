import logging
import time

import streamlit as st
from matplotlib import pyplot as plt

from examples.streamlit_app import render_prev_next
from examples.utils.animated_sliding_window_stream import AnimatedSlidingWindowStream
from examples.utils.plots import plot_audio, plot_spectrogram
from examples.utils.streamlit_worker import await_running_thread, run_managed_thread
from torchstream.sequence.sequence import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.title("3. Streaming BigVGAN")

"""
In the example #1, we mentioned that mel spectrograms are a common way to represent audio data for neural networks.
Mel spectrograms are cheap to compute, but hard to invert back to audio. This is called vocoding, and modern 
high quality vocoders often are neural networks. 

BigVGAN is a state of the art neural vocoder, part of many modern speech synthesis pipelines. For a text-to-speech 
system, streaming it is essential to reduce latency. For a speech-to-speech system, streaming it allows for 
live voice conversion or live speech denoising.

Let's load the model and a sample input:
"""


fig, ax = plt.subplots(figsize=(10, 2.5))

with st.echo():
    import torch

    from examples.resources.bigvgan.bigvgan import BigVGAN
    from examples.resources.bigvgan.meldataset import get_mel_spectrogram
    from examples.utils.audio import load_audio
    from examples.utils.download import download_file_cached

    device = "cuda" if torch.cuda.is_available() else "cpu"

    @st.cache_resource
    def load_bigvgan() -> BigVGAN:
        model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x")
        model.remove_weight_norm()
        model = model.eval().to(device)
        return model

    model = load_bigvgan()

    # Load an audio file at the model's samplerate
    MP3_URL = "https://d38nvwmjovqyq6.cloudfront.net/va90web25003/companions/ws_smith/32%20Speaking%20The%20Text%20As%20A%20Dramatic%20Reading.mp3"
    local_audio_path = download_file_cached(MP3_URL)
    wave, sample_rate = load_audio(
        local_audio_path,
        sample_rate=model.h.sampling_rate,
    )

    # Compute the mel spectrogam input
    mel = get_mel_spectrogram(
        torch.from_numpy(wave).unsqueeze(0),
        model.h,
    ).to(device)

st.write("##### Input audio & spectrogram")
st.audio(wave, sample_rate=sample_rate)
st.caption("Source: https://global.oup.com/us/companion.websites/9780195300505/audio/audio_samples/, sample 32")

plot_spectrogram(ax, mel[0], is_log=True, vmin=None, vmax=None)
st.pyplot(fig)

"""
And now run inference with it:
"""

start_time = time.perf_counter()
with st.echo():
    with torch.inference_mode():
        # mel is a (1, M, T_frames) shaped float32 tensor
        # wav_out is a (1, 1, T_samples) shaped float32 tensor
        wav_out = model(mel)

inference_time = time.perf_counter() - start_time

wav_out = wav_out.cpu().flatten()
st.write("##### Output audio")
st.audio(wav_out.numpy(), sample_rate=sample_rate)
st.write(f"_{len(wav_out) / sample_rate:.2f} seconds of audio generated in {inference_time:.2f} seconds._")

"""
It should sound the same as our input. We've recovered a great approximation of it from its mel spectrogram.

Let's proceed with streaming:
"""

with st.echo():
    from torchstream import SeqSpec

    # Mel spectrogram input
    in_spec = SeqSpec(1, model.h.num_mels, -1, device=device)
    # Audio waveform output
    out_spec = SeqSpec(1, 1, -1, device=device)

st.code("""
from torchstream import find_sliding_window_params
import logging

logging.basicConfig(level=logging.INFO)
        
sli_params = find_sliding_window_params(
    model,
    in_spec,
    out_spec,
    # BigVGAN produces outputs in the audio domain with a large receptive field, 
    # so the solver reaches the limit 100,000 on the input/output size while 
    # searching for a solution. We can safely increase it tenfold here.
    max_in_out_seq_size=1_000_000,
)[0]
""")


def find_sli_params_and_print(*args, **kwargs):
    sols = find_sliding_window_params(*args, **kwargs)

    logger.info("-----------------\n")
    for i, sol in enumerate(sols):
        logger.info(f"Solution #{i + 1}: {sol}")


run_managed_thread(
    func=find_sli_params_and_print,
    run_id="run1",
    job_id="bigvgan_demo",
    func_kwargs=dict(
        trsfm=model,
        in_spec=in_spec,
        out_spec=out_spec,
        max_in_out_seq_size=1_000_000,
    ),
    log_height=500,
)
await_running_thread()


"""
This time around we managed to find a solution without hitting any snags. Just like in the previous example, we 
converge to a rather complex solution. Going through the forward function of bigvgan involves the computation 
of **about 280 torch primitives** (convs, tconvs, activations, residual sums, ...). So it's quite magical to be able to 
reduce this big model to just a sliding window algorithm!

For a transform of this complexity we have a myriad of possible sliding window parameters that correspond to it, 
but still **all with the same input/output size relation, output delay and input context size**. Due to the 
randomness of the solver, you might even have obtained a different solution than the one hardcoded below.

There are [many variations of hyperparameters for BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x#pretrained-models), 
**each will have their own set of sliding window parameters**. But you only need to compute them once with the solver, 
and you can then store them alongside the hyperparameters.

"""

with st.echo():
    from torchstream import SlidingWindowParams

    sli_params = SlidingWindowParams(
        kernel_size_in=75,
        stride_in=1,
        left_pad=37,
        right_pad=37,
        kernel_size_out=314,
        stride_out=256,
        left_out_trim=29,
        right_out_trim=29,
    )

    st.code(
        f"-> min/max overlap: {[sli_params.streaming_context_size, sli_params.streaming_context_size + sli_params.stride_in - 1]}\n"
        + f"-> min/max output delay: {list(sli_params.output_delay_bounds)}\n"
        + f"-> in/out size relation: {sli_params.in_out_size_rel_repr}"
    )

"""
Below is another interactive demo of the streaming so you can visualize it:
"""

st.code("""
from torchstream import SlidingWindowStream
        
stream = SlidingWindowStream(model, sli_params, in_spec, out_spec)
for chunk in stream.forward_in_chunks_iter(mel, chunk_size=chunk_size):
    ...
""")

with st.container(border=True):
    total_seconds = len(wave) / sample_rate
    min_slice_seconds = 0.1
    start_sec, end_sec = st.slider(
        "Select the audio segment",
        0.0,
        total_seconds,
        (0.0, 5.0),
        step=0.1,
        format="%.1fs",
    )
    if end_sec - start_sec < min_slice_seconds:
        end_sec = min(total_seconds, start_sec + min_slice_seconds)
        if end_sec - start_sec < min_slice_seconds:
            start_sec = max(0.0, end_sec - min_slice_seconds)
    start_sample = int(start_sec * sample_rate)
    end_sample = max(start_sample + int(min_slice_seconds * sample_rate), int(end_sec * sample_rate))

    wave_slice = wave[start_sample:end_sample]
    mel = get_mel_spectrogram(torch.from_numpy(wave_slice).unsqueeze(0), model.h).to(device)
    with torch.inference_mode():
        wav_out = model(mel).cpu().flatten().numpy()

    def build_stream(chunk_size: int) -> AnimatedSlidingWindowStream:
        stream_obj = AnimatedSlidingWindowStream(model, sli_params, in_spec, out_spec)
        stream_obj.forward_in_chunks(mel, chunk_size=chunk_size)
        return stream_obj

    chunk_size = st.slider(
        "Set the streaming chunk size",
        sli_params.min_input_size,
        mel.size(2) // 2,
        value=mel.size(2) // 5,
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
        fig.suptitle(f"Streaming BigVGAN - Step {step_idx + 1}/{len(stream.step_history)}", fontsize=18)

        # Input plot
        plot_spectrogram(axs[0], mel[0], is_log=True, vmin=None, vmax=None)

        # Sync output plot
        plot_audio(axs[2], wav_out, sample_rate)

        def out_plot_fn(ax, data):
            plot_audio(ax, data.cpu().flatten().numpy(), sample_rate=sample_rate)

        stream.plot_step(step_idx, *axs, out_plot_fn=out_plot_fn)
        plot_placeholder.pyplot(fig)

    stream_step_fragment()

"""
BigVGAN is a high stack of convolutions with dilation, leading to large kernels. Their individual receptive field 
add up and multiply through the upsampling layers (=transposed convolutions with stride>1), leading to a very large 
output delay. It is of 9501 samples here, which is about 400ms at 24kHz. This not only **wasteful compute**¹ but also 
a **noticeable latency**. Even if you were to stream frame-by-frame² instantaneously, you would still have a 400ms 
delay between your input spectrogram and your output audio. 

Yet you can see from the above images that the trimmed portion of the output is quite faithful to the original audio! 
We should be able to keep some of that trimmed output.

The sliding window parameter solver finds the original parameters of the transform, which allows us to stream it 
**exactly**. Aside from minor differences in computation introduced by optimizations and floating point rounding, **the 
output you get from a `SlidingWindowStream`** with the correct parameters **will be identical to the non-streamed 
transform**.

Hence, you can significantly reduce the output delay at the cost of some output fidelity. TorchStream does not yet 
expose functions to enable this, but it is a planned feature.
"""
st.caption(
    "¹ this is _technically_ wasteful compute. Depending on your choices of chunk size and your benchmarks, "
    "the additional compute might end up being negligible"
)
st.caption(
    "² i.e. with input chunks of size 1. This is possible because the parameters indicate a minimum input size of 1"
)

"""
### Up next
We've found the sliding window parameters of a deep neural network and made it streamable. The following 
example will target a more complex architecture, with some non-sliding-window components.
"""

render_prev_next(__file__)
