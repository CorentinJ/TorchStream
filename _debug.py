import logging

import numpy as np
import torch
import torchaudio

from torchstream import (
    SeqSpec,
)
from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


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


print(
    find_sliding_window_params(
        lambda x: get_spectrogram(x, 1000, n_fft=256, center=False, hop_size=16),
        SeqSpec(-1, dtype=np.float32),
        SeqSpec(120, -1),
        zero_size_exception_signatures=DEFAULT_ZERO_SIZE_EXCEPTIONS + [(RuntimeError, "expected 0 < n_fft <")],
        max_equivalent_sols=10,
    )
)
