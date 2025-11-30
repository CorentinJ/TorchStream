import inspect
import logging

import librosa
import librosa.core
import numpy as np
import streamlit as st

from dev_tools.tracing import log_tracing_profile
from examples.utils.audio import load_audio
from examples.utils.download import download_file_cached
from examples.utils.streamlit_worker import run_managed_thread
from torchstream import SeqSpec, find_sliding_window_params
from torchstream.patching.call_intercept import intercept_calls
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

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
The solver takes any function that transforms sequential data (torch tensors, numpy arrays) into other sequential 
data. The data can be any shape or data type, it can be audio, video, text, etc... It can also be a combination of 
multiple arrays (more on this in example #4).

Then:
1. It probes the function with a randomly generated input to see if it behaves correctly, until it finds a valid 
input-output pair.
2. It infers the input size to output size relationship of the function by forwarding multiple inputs of different 
sizes.
3. It finds sliding window parameters that would explain the observed inputs and outputs, and it verifies that they 
are correct by generating specific inputs and checking the outputs.

Let's test it on a simple example. We'll write a moving average function with window size and stride as parameters.
"""

code_placeholder = st.empty()

left_col, right_col = st.columns([0.2, 0.8])

with left_col:
    win_size = st.slider("Window size", min_value=1, max_value=10, value=3)
    stride_in = st.slider("Input stride", min_value=1, max_value=10, value=2)


def moving_average(x: np.ndarray) -> np.ndarray:
    out = []
    for start_idx in range(0, len(x), stride_in):
        window = x[start_idx : start_idx + win_size]
        if len(window) < win_size:
            break
        out.append(np.mean(window))
    return np.array(out)


code_placeholder.code(
    inspect.getsource(moving_average)
    + """
find_sliding_window_params(
    moving_average,
    # Input spec is the same as output spec, no need to specify it twice
    in_spec=SeqSpec(-1, dtype=np.float32),
    # A few cases have multiple equivalent solutions e.g. (win_size=2, stride_in=1)
    max_equivalent_sols=3,
)
"""
)


def find_sli_params_and_print(*args, **kwargs):
    sols = find_sliding_window_params(*args, **kwargs)

    logger.info("-----------------\n")
    for i, sol in enumerate(sols):
        logger.info(f"Solution #{i + 1}: {sol}")


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
The solver quickly finds one or multiple solutions, including the exact parameters we used. When it finds multiple 
solutions, **they are equivalent** in the sense that all produce the same input to output mapping. It does not matter 
which one you use down the line, they will all work the same. By default `max_equivalent_sols` is set to 1. 
"""

left_col, right_col = st.columns([0.5, 0.5])
with left_col:
    with st.echo():
        params = SlidingWindowParams(
            kernel_size_in=2,
        )

    st.code(
        "Mapping for input of size 10:\n"
        + "\n".join(
            f"   in_range {list(in_range)} -> out_range {list(out_range)}"
            for in_range, out_range in params.iter_bounded_kernel_map(10)
        )
    )

with right_col:
    with st.echo():
        params = SlidingWindowParams(
            kernel_size_out=2,
            left_out_trim=1,
            right_out_trim=1,
        )

    st.code(
        "Mapping for input of size 10:\n"
        + "\n".join(
            f"   in_range {list(in_range)} -> out_range {list(out_range)}"
            for in_range, out_range in params.iter_bounded_kernel_map(10)
        )
    )


quit()
"""
### Audio resampling

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
    wave_8khz = librosa.core.resample(wave_48khz, orig_sr=48000, target_sr=8000)

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
microphone.

Again, a naive approach would lead to awful audio artifacts that will make anyone turn away from your application.
"""

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

            def resample_fn(y):
                with intercept_calls("librosa.util.utils.valid_audio", lambda wav: True):
                    return librosa.core.resample(
                        y, orig_sr=sample_rate_in, target_sr=sample_rate_out, res_type=res_algo
                    )

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
