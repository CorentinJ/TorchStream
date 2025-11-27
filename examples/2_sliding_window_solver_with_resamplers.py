import logging

import librosa
import librosa.core
import numpy as np
import streamlit as st

from dev_tools.tracing import log_tracing_profile
from torchstream import SeqSpec, find_sliding_window_params
from torchstream.patching.call_intercept import intercept_calls

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.subheader("2. The Sliding Window Parameter Solver")


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
        for sample_rate_in, sample_rate_out in [(16000, 48000), (32000, 16000)]:

            def resample_fn(y):
                with intercept_calls("librosa.util.utils.valid_audio", lambda wav: True):
                    return librosa.core.resample(
                        y, orig_sr=sample_rate_in, target_sr=sample_rate_out, res_type=res_algo
                    )

            sols = find_sliding_window_params(
                resample_fn,
                SeqSpec(-1, dtype=np.float32),
                zero_size_exception_signatures=ZERO_SIZE_EXCEPTION_SIGNATURES,
            )
            results.append((res_algo, sample_rate_in, sample_rate_out, sols[0]))

print("----")
for res_algo, sr_in, sr_out, sol in results:
    print(f"Resampling with {res_algo} from {sr_in} to {sr_out}:\n{sol}")
