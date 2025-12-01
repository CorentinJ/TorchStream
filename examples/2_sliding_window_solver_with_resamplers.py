import logging

import librosa
import librosa.core
import numpy as np
import streamlit as st
import torch
from torch import nn

from dev_tools.tracing import log_tracing_profile
from examples.utils.audio import load_audio
from examples.utils.download import download_file_cached
from examples.utils.streamlit_worker import run_managed_thread
from torchstream import SeqSpec, SlidingWindowParams, find_sliding_window_params, intercept_calls
from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.subheader("2. The Sliding Window Parameter Solver")

"""
In the first example we mentioned that TorchStream can automatically determine the sliding window parameters of a 
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
The solver takes any function that transforms sequential data (torch tensors, numpy arrays) into other sequential 
data. The data can be any shape or data type, it can be audio, video, text, etc... It can also be a combination of 
multiple arrays (more on this in example #4).

Then:
1. It probes the function with a randomly generated input to see if it behaves correctly, until it finds a valid 
input-output pair.
2. It infers the input size to output size relationship of the function by forwarding multiple inputs of different 
sizes.
3. It finds sliding window parameters that would explain the observed inputs and outputs, and it verifies that they 
are correct by generating new specific inputs and checking their outputs.

Let's test it on a simple example. We'll write a moving average function with window size and stride as parameters.
"""


def find_sli_params_and_print(*args, **kwargs):
    sols = find_sliding_window_params(*args, **kwargs)

    logger.info("-----------------\n")
    for i, sol in enumerate(sols):
        logger.info(f"Solution #{i + 1}: {sol}")


@st.fragment
def moving_average_demo():
    print("RERUN")
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
from torchstream import SeqSpec, SlidingWindowParams, find_sliding_window_params

logging.basicConfig(level=logging.INFO)

def moving_average(x: np.ndarray) -> np.ndarray:
    out = []
    for start_idx in range(0, len(x), stride_in):
        window = x[start_idx : start_idx + win_size]
        if len(window) < win_size:
            break
        out.append(np.mean(window))
    return np.array(out)
    
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


moving_average_demo()

"""
It does not matter which of the solver's solutions you use down the line, they will all work the same. The solver 
never returns suboptimal or incorrect solutions. Hence by default, `max_equivalent_sols` is set to 1. 
"""

"""
**NaN trick**: the solver works using a simple trick. We rely on NaN propagation¹ to understand how the transform 
maps input indices to output indices. Python, numpy and torch will output a NaN in virtually every operation that 
has a NaN for operand. For example:
"""
st.caption("¹ Defined in the IEEE 754 Standard for Floating-Point Arithmetic")

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
If you go into the resample function, you'll see that this method is `valid_audio()`. In practice, you can understand 
these types of issues by **going through stack traces** or ideally by **stepping into the transform with a debugger**.
"""

with st.container(border=True):
    """
    It is a frequent occurrence when trying to make a transform streamable that one (possibly deeply) nested function 
    will get in your way. TorchStream offers **monkey patching utilities** to get you past these hurdles without having 
    to rewrite code. Once you've figured out how to stream your model, you can usually do without monkey patching.
    """


"""
Use `torchstream.intercept_calls()` to replace the function with a no-op that always returns True. We only need to do 
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

Therefore the solver's job is to swallow these exceptions and work as if the transform gave a **zero-sized output**. 
Because there is no universal exception type for "zero-sized output", you can provide its signature as a tuple 
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
And we're off
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
)


quit()

RESAMPLING_ALGOS = [
    "kaiser_best",
    "kaiser_fast",
    "soxr_qq",
    "polyphase",
]

ZERO_SIZE_EXCEPTION_SIGNATURES = [
    # Raised by kaiser resamplers
    (ValueError, "is too small to resample from"),
]


results = []
with log_tracing_profile("solver"):
    for res_algo in RESAMPLING_ALGOS:
        for sample_rate_in, sample_rate_out in [(16000, 48000), (16000, 32000)]:
            try:
                sols = find_sliding_window_params(
                    resample_fn,
                    SeqSpec(-1, dtype=np.float32),
                    zero_size_exception_signatures=ZERO_SIZE_EXCEPTION_SIGNATURES,
                )
            except RuntimeError as e:
                results.append((res_algo, sample_rate_in, sample_rate_out, f"Failed to solve: {e}"))
            else:
                results.append((res_algo, sample_rate_in, sample_rate_out, sols[0]))

print("----")
for res_algo, sr_in, sr_out, sol in results:
    print(f"Resampling with {res_algo} from {sr_in} to {sr_out}:\n{sol}")


@st.fragment
def resampling_solver_demo1():
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
    )
