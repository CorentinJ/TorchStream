import logging

import librosa
import librosa.core
import numpy as np

from torchstream import DEFAULT_ZERO_SIZE_EXCEPTIONS, intercept_calls
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def resample_trsfm(x: np.ndarray) -> np.ndarray:
    with intercept_calls("librosa.util.utils.valid_audio", handler_fn=lambda wav: True):
        return librosa.core.resample(x, orig_sr=48000, target_sr=16000, res_type="kaiser_best")


find_sliding_window_params(
    resample_trsfm,
    in_spec=SeqSpec(-1, dtype=np.float32),
    zero_size_exception_signatures=DEFAULT_ZERO_SIZE_EXCEPTIONS
    + [
        (ValueError, "is too small to resample from"),
    ],
)
