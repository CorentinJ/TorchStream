import subprocess

import numpy as np
import streamlit as st
import torch


class WebRadioStream:
    """
    Streams audio from a web radio URL (using ffmpeg), yielding chunks as PyTorch tensors.

    Much easier to interface than the microphone for the purpose of live demos...
    """

    def __init__(
        self,
        url: str = "https://icecast.radiofrance.fr/fip-midfi.mp3",
        sample_rate: int = 48000,
        chunk_duration_ms: int = 50,
    ) -> None:
        self.url = url
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.channels = 1
        self.samples_per_chunk = max(1, int(sample_rate * chunk_duration_ms / 1000))
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            url,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(self.channels),
            "-ar",
            str(sample_rate),
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.samples_per_chunk * self.channels * 2,
        )
        if self.proc.stdout is None:
            raise RuntimeError("Could not open ffmpeg stdout for radio stream.")

    def read(self) -> torch.Tensor:
        if self.proc.poll() is not None:
            raise StopIteration("Radio process terminated.")

        bytes_needed = self.samples_per_chunk * self.channels * 2
        data = self.proc.stdout.read(bytes_needed)
        if not data or len(data) < bytes_needed:
            raise StopIteration("Radio stream ended.")

        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio = torch.from_numpy(audio).view(-1, self.channels)
        if self.channels > 1:
            audio = audio.mean(dim=1)
        else:
            audio = audio.squeeze(1)
        return audio

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __del__(self) -> None:
        self.close()


def streamlit_ensure_web_radio_stream(
    url: str = "https://icecast.radiofrance.fr/fip-midfi.mp3",
    sample_rate: int = 48000,
    chunk_duration_ms: int = 50,
) -> WebRadioStream:
    """
    Instantiates or retrieves a WebRadioStream object stored in Streamlit's session state.
    """
    stream = st.session_state.get("_radio_stream")
    if stream is not None:
        if stream.url != url or stream.sample_rate != sample_rate or stream.chunk_duration_ms != chunk_duration_ms:
            stream.close()
            stream = None

    if stream is None:
        try:
            stream = WebRadioStream(url, sample_rate, chunk_duration_ms)
        except FileNotFoundError:
            st.error("ffmpeg executable not found. Install ffmpeg to decode the radio stream.")
            st.stop()
        except RuntimeError as exc:
            st.error(f"Could not open radio stream: {exc}")
            st.stop()

        st.session_state["_radio_stream"] = stream

    return stream
