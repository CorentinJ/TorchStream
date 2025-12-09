import inspect
import logging
from functools import partial

import librosa
import librosa.core
import numpy as np
import streamlit as st
import torch
from matplotlib import pyplot as plt
from torch import nn

from examples.streamlit_app import render_prev_next
from examples.utils.animated_sliding_window_stream import AnimatedSlidingWindowStream
from examples.utils.audio import load_audio
from examples.utils.download import download_file_cached
from examples.utils.plots import plot_audio
from examples.utils.streamlit_worker import await_running_thread, run_managed_thread
from torchstream import SeqSpec, SlidingWindowParams, find_sliding_window_params, intercept_calls
from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS
from torchstream.sequence.sequence import Sequence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.title("2. The Sliding Window Parameters Solver (with audio resamplers)")

"""
In the first example we mentioned that TorchStream can **automatically** determine the sliding window parameters of a 
given transform, but did not exploit it. 

This second example will cover transforms for which the sliding window parameters are not disclosed and go into 
the details of how TorchStream proceeds to find them.
"""

with st.container(border=True):
    """
    You must understand how the sliding window solver works under the hood to use it in your own applications.
    """

"""
### Overview
The solver takes any function **that transforms sequential data** (torch tensors, numpy arrays) into **other sequential 
data**. The data can be any shape or data type, it can be audio, video, text, etc... It can also be a combination of 
multiple arrays (covered in example #4).

Then:
1. **It probes the function** with a randomly generated input to see if it behaves correctly, until it finds a valid 
input-output pair.
2. **It infers the input size to output size relationship** of the function by forwarding multiple inputs of different 
sizes.
3. **It finds sliding window parameters** that would explain the observed inputs and outputs, and it verifies that they 
are compatible by generating new specific inputs and checking their outputs.

Let's test it on a simple example. We'll write a moving average function with window size and stride as parameters.
"""


def find_sli_params_and_print(*args, **kwargs):
    try:
        sols = find_sliding_window_params(*args, **kwargs)
    except RuntimeError as e:
        logger.info("-----------------\n")
        logger.info(f"Solver failed with error: {e}")
        return

    logger.info("-----------------\n")
    for i, sol in enumerate(sols):
        logger.info(f"Solution #{i + 1}: {sol}")


code_placeholder = st.empty()

left_col, right_col = st.columns([0.2, 0.8])

with left_col:
    win_size = st.slider("Window size", min_value=1, max_value=10, value=3)
    stride_in = st.slider("Input stride", min_value=1, max_value=10, value=2)
    st.caption(
        "Note: the solver will fail with a stride larger than the window size. These would be invalid parameters, "
        "skipping entirely over some inputs."
    )


def moving_average(x: np.ndarray) -> np.ndarray:
    out = []
    for start_idx in range(0, len(x), stride_in):
        window = x[start_idx : start_idx + win_size]
        if len(window) < win_size:
            break
        out.append(np.mean(window))
    return np.array(out)


code_placeholder.code(
    """
import logging

import numpy as np

from torchstream import SeqSpec, find_sliding_window_params

# The solver emits INFO level logs, enable them to see its progress
logging.basicConfig(level=logging.INFO)


"""
    + inspect.getsource(moving_average)
    + """
    
find_sliding_window_params(
    moving_average,
    # Input spec is the same as output spec, no need to specify it twice
    in_spec=SeqSpec(-1, dtype=np.float32),
    # Yield up to 3 equivalent solutions (default is 1)
    max_equivalent_sols=3,
)
"""
)

with right_col:
    run_managed_thread(
        func=find_sli_params_and_print,
        run_id=f"run_{win_size}_{stride_in}",
        job_id="moving_average_demo",
        func_kwargs=dict(
            trsfm=moving_average,
            in_spec=SeqSpec(-1, dtype=np.float32),
            max_equivalent_sols=3,
            max_hypotheses=30,
        ),
    )

"""
**Multiplicity of solutions**: The solver quickly finds one or multiple solutions, including the exact parameters we used. When it finds multiple 
solutions, **they are equivalent** in the sense that all produce the same input to output mapping. For instance, these 
two are equivalent:
"""


def print_map(params: SlidingWindowParams, input_size: int):
    # Get input ranges for each window
    win_inputs = [in_range for in_range, _ in params.iter_bounded_kernel_map(input_size)]

    # Map each output index to every input indices that it sees
    # TODO: specialized methods for this in SlidingWindowParams? This is the receptive field, it's important
    input_ranges_by_output_idx = []
    for out_start, out_end, windows in params.get_inverse_kernel_map(input_size):
        min_in_idx = input_size
        max_in_idx = 0
        for win_idx, _, _ in windows:
            win_in_range = win_inputs[win_idx]
            min_in_idx = min(min_in_idx, win_in_range[0])
            max_in_idx = max(max_in_idx, win_in_range[1] - 1)
        input_range = [min_in_idx, max_in_idx + 1]

        input_ranges_by_output_idx.extend([input_range] * (out_end - out_start))

    st.code(
        f"Mapping for input of size {input_size}:\n"
        + "[Output idx]             [Input range]\n"
        + "\n".join(
            f"    [{i}]     computed from   {input_ranges_by_output_idx[i]}"
            for i in range(len(input_ranges_by_output_idx))
        )
    )


left_col, right_col = st.columns([0.5, 0.5])
with left_col:
    with st.echo():
        params = SlidingWindowParams(
            kernel_size_in=2,
        )

    print_map(params, input_size=10)

with right_col:
    with st.echo():
        params = SlidingWindowParams(
            kernel_size_out=2,
            left_out_trim=1,
            right_out_trim=1,
        )

    print_map(params, input_size=10)


"""
It does not matter which of the solver's solutions you use down the line, they will all work the same. The solver 
never returns suboptimal or incorrect solutions. Hence by default, `max_equivalent_sols` is set to 1. 
"""

"""
**NaN trick**: the solver works using a simple trick. We rely on NaN propagation¹ to understand how the transform 
maps input indices to output indices. Python, numpy and torch will output a NaN in virtually every operation that 
has a NaN for operand. For example:
"""
st.caption("¹ A convention defined in the IEEE 754 Standard for Floating-Point Arithmetic")

with st.echo():
    x = torch.tensor([[[1.0, 2.0, 3.0, float("nan"), 5.0, 6.0, 7.0]]])
    nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, dilation=2)(x)
st.code(nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, dilation=2)(x).detach().tolist())

"""
Your transform must accept NaNs as inputs for this to work. Below we'll explore workarounds for when that is not the case.
"""

"""
### Example with audio resampling

To resample an audio signal means to modify it to be as if it was recorded at a different sample rate. Common 
sample rates range from 8kHz (telephone quality), i.e. 8000 audio samples per second, to 48kHz (professional audio 
quality).

The python library librosa offers a variety of audio resampling algorithms with different performance and quality 
trade-offs.
"""

MP3_URL = "https://d38nvwmjovqyq6.cloudfront.net/va90web25003/companions/ws_smith/32%20Speaking%20The%20Text%20As%20A%20Dramatic%20Reading.mp3"
with st.echo():
    import librosa

    from examples.utils.audio import load_audio
    from examples.utils.download import download_file_cached

    local_audio_path = download_file_cached(MP3_URL)
    wave_48khz, _ = load_audio(local_audio_path, sample_rate=48000)
    wave_8khz = librosa.core.resample(wave_48khz, orig_sr=48000, target_sr=8000, res_type="kaiser_best")

st.audio(wave_48khz, sample_rate=48000)
st.caption("Original audio at 48kHz")

st.audio(wave_8khz, sample_rate=8000)
st.caption("Downsampled audio at 8kHz")

"""
This operation changes the size of the audio signal (here by a factor of 6):
"""
st.code(
    f"wave_48khz: {len(wave_48khz):,} sized {wave_48khz.dtype} numpy array\n"
    f" wave_8khz: {len(wave_8khz):,} sized {wave_8khz.dtype} numpy array",
)

"""
It is common to need to stream this resampling operation in **real-time**, for example when receiving audio data from a
microphone. Here again, a naive approach would lead to awful audio artifacts that would degrade the quality of your 
application.

Let's involve the solver:
"""


st.code("""
def resample_trsfm(x: np.ndarray) -> np.ndarray:
    return librosa.core.resample(x, orig_sr=48000, target_sr=8000, res_type="kaiser_best")

sols = find_sliding_window_params(
    resample_trsfm,
    SeqSpec(-1, dtype=np.float32),
)""")
st.exception(librosa.util.exceptions.ParameterError("Audio buffer is not finite everywhere"), width=400)

"""
And we've hit our first snag. Librosa has an internal check to verify that input audio is valid, not containing NaNs. 
If you go into the resample function, you'll see that this method is `valid_audio()`. In practice, you can figure 
out these types of issues by **going through stack traces** or ideally by **stepping into the transform with a 
debugger**.
"""

with st.container(border=True):
    """
    It is a frequent occurrence when trying to make a transform streamable that one (possibly deeply) nested function 
    will get in your way. TorchStream offers **monkey patching utilities** to get you past these hurdles without having 
    to rewrite code. Once you've figured out how to stream your model, you can usually do without monkey patching.
    """


"""
Use `torchstream.intercept_calls()` to replace the function with a no-op that always returns `True`. We only need to do 
this during the solver's execution.
"""

st.code("""
from torchstream import intercept_calls

def resample_trsfm(x: np.ndarray) -> np.ndarray:
    with intercept_calls("librosa.util.utils.valid_audio", handler_fn=lambda wav: True):
        return librosa.core.resample(x, orig_sr=48000, target_sr=8000, res_type="kaiser_best")
        
find_sliding_window_params(
    resample_trsfm,
    in_spec=SeqSpec(-1, dtype=np.float32),
)
""")
st.exception(ValueError("Input signal length=1 is too small to resample from 48000->8000"), width=400)

"""
Second snag: we get an Exception when forwarding an input size that is too small. The solver _will_ provide inputs 
that are too small for the given transform to produce any output, because finding **the minimum input size** of the 
transform is part of its job.

Therefore the solver must swallow these exceptions and work as if the transform gave a **zero-sized output**. 
Because there is no universal exception type for "input is too small", you can provide its signature as a tuple 
`(exception_type, message_substring)` like so:

"""

st.code("""
from torchstream import DEFAULT_ZERO_SIZE_EXCEPTIONS

find_sliding_window_params(
    resample_trsfm,
    in_spec=SeqSpec(-1, dtype=np.float32),
    zero_size_exception_signatures=DEFAULT_ZERO_SIZE_EXCEPTIONS + [
        (ValueError, "is too small to resample from"),
    ],
)
""")


"""
And we're off:
"""


def resample_trsfm(x: np.ndarray) -> np.ndarray:
    with intercept_calls("librosa.util.utils.valid_audio", handler_fn=lambda wav: True):
        return librosa.core.resample(x, orig_sr=48000, target_sr=8000, res_type="kaiser_best")


run_managed_thread(
    func=find_sli_params_and_print,
    run_id="run1",
    job_id="resample_demo1",
    func_kwargs=dict(
        trsfm=resample_trsfm,
        in_spec=SeqSpec(-1, dtype=np.float32),
        zero_size_exception_signatures=DEFAULT_ZERO_SIZE_EXCEPTIONS
        + [
            (ValueError, "is too small to resample from"),
        ],
    ),
    log_height=500,
)

"""
The solution here is certainly more complex than the ones with the moving average example. It would have been difficult 
for the developer of the resampling function to find and document these values. You can imagine how hard it gets with 
neural networks!

Let's visualize the streaming of this resampling operation:
"""


with st.container(border=True):
    with st.container(border=True):
        sli_params = SlidingWindowParams(
            kernel_size_in=11,
            stride_in=6,
            left_pad=5,
            right_pad=5,
            kernel_size_out=99,
            stride_out=1,
            left_out_trim=49,
            right_out_trim=49,
            min_input_size=6,
        )
        st.code(
            str(sli_params)
            + f"\n-> min/max overlap: {[sli_params.streaming_context_size, sli_params.streaming_context_size + sli_params.stride_in - 1]}\n"
            + f"-> min/max output delay: {list(sli_params.output_delay_bounds)}\n"
            + f"-> in/out size relation: {sli_params.in_out_size_rel_repr}"
        )

    total_seconds = len(wave_48khz) / 48000
    min_slice_seconds = 0.01
    start_sec, end_sec = st.slider(
        "Select the audio segment",
        0.0,
        1.0,
        (0.35, 0.4),
        step=0.001,
        format="%.2fs",
    )
    if end_sec - start_sec < min_slice_seconds:
        end_sec = min(total_seconds, start_sec + min_slice_seconds)
        if end_sec - start_sec < min_slice_seconds:
            start_sec = max(0.0, end_sec - min_slice_seconds)
    start_sample = int(start_sec * 48000)
    end_sample = max(start_sample + int(min_slice_seconds * 48000), int(end_sec * 48000))

    wave_48khz_slice = wave_48khz[start_sample:end_sample]

    def build_stream(chunk_size: int) -> AnimatedSlidingWindowStream:
        stream_obj = AnimatedSlidingWindowStream(
            resample_trsfm,
            sli_params,
            SeqSpec(-1, dtype=np.float32),
        )
        stream_obj.forward_in_chunks(Sequence(wave_48khz_slice, seq_dim=0), chunk_size=chunk_size)
        return stream_obj

    chunk_size = st.slider(
        "Set the streaming chunk size",
        sli_params.min_input_size,
        len(wave_48khz_slice) // 2,
        value=len(wave_48khz_slice) // 5,
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
        fig.suptitle(f"Streaming librosa.resample - Step {step_idx + 1}/{len(stream.step_history)}", fontsize=18)

        # Input plot
        plot_audio(axs[0], wave_48khz_slice, 48000, tick_label_offset_s=start_sec, as_scatter=True, with_end_tick=False)

        # Sync output plot
        plot_audio(axs[2], resample_trsfm(wave_48khz_slice), 8000, as_scatter=True, with_end_tick=False)

        stream.plot_step(
            step_idx, *axs, out_plot_fn=partial(plot_audio, sample_rate=8000, as_scatter=True, with_end_tick=False)
        )
        plot_placeholder.pyplot(fig)

    stream_step_fragment()

"""
The waveforms are plotted as scatter instead of lines to better differentiate between the higher and lower sample 
rates. The plot is also quite zoomed in on the audio so you can see the output delay. It goes up to 
50, meaning that you will get at least 50/8000 = 6.25ms of delay with this streaming implementation (exact latency and 
output delay will be covered in another example).

Feel free to play in a separate script with other resampling algorithms by changing the `res_type` parameter 
(soxr_qq, polyphase, ...). Not all algorithms behave as sliding window transforms. Resampling between the 
44.1 kHz family vs the 48 kHz family will also not be modelable as sliding window transforms given the large 
discrepancy in common factors.

### Up next
You've seen how to automatically find the sliding window parameters of a simple and of a more complex transform.

In the following examples we'll attack real world neural networks.
"""


render_prev_next(__file__)

await_running_thread()
