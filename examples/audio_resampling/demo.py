import logging
from unittest.mock import patch

import librosa
import librosa.core
import numpy as np

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def my_transform(y):
    with patch("librosa.util.utils.valid_audio", return_value=True):
        return librosa.core.resample(y, orig_sr=32000, target_sr=16000)


find_sliding_window_params(my_transform, SeqSpec((-1), dtype=np.float32))
