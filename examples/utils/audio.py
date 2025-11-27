from pathlib import Path
from typing import Tuple

import audioread
import librosa
import numpy as np


def load_audio(path: Path, sample_rate=None) -> Tuple[np.ndarray, int]:
    with audioread.audio_open(str(path)) as f:
        orig_sample_rate = f.samplerate
        channels = f.channels
        pcm_chunks = [np.frombuffer(buf, dtype=np.int16) for buf in f]

    if not pcm_chunks:
        return np.empty(0, dtype=np.float32), orig_sample_rate

    audio = np.concatenate(pcm_chunks)

    if channels > 1:
        trimmed = (audio.size // channels) * channels
        audio = audio[:trimmed].reshape(-1, channels).mean(axis=1)

    wave = audio.astype(np.float32) / 32768.0

    if sample_rate:
        wave = librosa.resample(wave, orig_sr=orig_sample_rate, target_sr=sample_rate)
        orig_sample_rate = sample_rate

    return wave, orig_sample_rate
