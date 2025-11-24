import subprocess
import threading
from queue import Queue

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
    ) -> None:
        self.url = url
        self.sample_rate = sample_rate
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
            "1",
            "-ar",
            str(sample_rate),
            "-",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError:
            st.error("ffmpeg executable not found. Install ffmpeg to decode the radio stream.")
            st.stop()

        if self.proc.stdout is None:
            raise RuntimeError("Could not open ffmpeg stdout for radio stream.")

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        self.buffer = Queue()

    def _reader_loop(self) -> None:
        while True:
            data = self.proc.stdout.read(4096)
            if not data:
                break
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            self.buffer.put(audio)

    def read(self) -> torch.Tensor:
        if not self.buffer.qsize():
            return torch.empty(0, dtype=torch.float32)
        audio = np.concatenate([self.buffer.get_nowait() for _ in range(self.buffer.qsize())])
        audio = torch.from_numpy(audio)
        return audio

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()

        self.thread.join()

    def __del__(self) -> None:
        self.close()


def streamlit_ensure_web_radio_stream(
    url: str = "https://icecast.radiofrance.fr/fip-midfi.mp3",
    sample_rate: int = 48000,
) -> WebRadioStream:
    """
    Instantiates or retrieves a WebRadioStream object stored in Streamlit's session state.
    """
    stream = st.session_state.get("_radio_stream")
    if stream is not None:
        if stream.url != url or stream.sample_rate != sample_rate:
            stream.close()
            stream = None

    if stream is None:
        stream = WebRadioStream(url, sample_rate)
        st.session_state["_radio_stream"] = stream

    return stream
