import logging
from unittest.mock import patch

import librosa
import librosa.core
import numpy as np

from dev_tools.tracing import log_tracing_profile
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


RESAMPLING_ALGOS = [
    dict(name="kaiser_best", zero_size_exception_types=(ValueError,)),
]


for res_dict in RESAMPLING_ALGOS:

    def resample_fn(y):
        with patch("librosa.util.utils.valid_audio", return_value=True):
            return librosa.core.resample(y, orig_sr=32000, target_sr=16000, res_type=res_dict["name"])

    with log_tracing_profile("solver"):
        sols = find_sliding_window_params(
            resample_fn,
            SeqSpec(-1, dtype=np.float32),
            zero_size_exception_types=res_dict["zero_size_exception_types"],
        )
        print(sols)
