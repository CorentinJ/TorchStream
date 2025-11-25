from pathlib import Path
from typing import Tuple

import audioread
import numpy as np


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    with audioread.audio_open(str(path)) as f:
        sample_rate = f.samplerate
        channels = f.channels
        pcm_chunks = [np.frombuffer(buf, dtype=np.int16) for buf in f]

    if not pcm_chunks:
        return np.empty(0, dtype=np.float32), sample_rate

    audio = np.concatenate(pcm_chunks)

    if channels > 1:
        trimmed = (audio.size // channels) * channels
        audio = audio[:trimmed].reshape(-1, channels).mean(axis=1)

    wave = audio.astype(np.float32) / 32768.0

    return wave, sample_rate
