import inspect
import logging
import multiprocessing as mp
import queue
import time
from typing import Optional, Tuple

import streamlit as st
import torchaudio

from torchstream import (
    SeqSpec,
    SlidingWindowStream,
    find_sliding_window_params,
    test_stream_equivalent,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class _QueueLogHandler(logging.Handler):
    """Send log records to a multiprocessing queue."""

    def __init__(self, log_queue: mp.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        try:
            self.log_queue.put(("log", msg))
        except Exception:
            pass


def run_solver(
    n_fft: int,
    center: bool,
    pad_mode: str,
    hop_length: Optional[int],
) -> Tuple[list, object]:
    """Compute sliding-window params for a given spectrogram configuration."""
    kwargs = dict(n_fft=n_fft, center=center, pad_mode=pad_mode)
    if hop_length is not None:
        kwargs["hop_length"] = hop_length

    transform = torchaudio.transforms.Spectrogram(**kwargs)

    in_spec = SeqSpec(-1)
    out_spec = SeqSpec(n_fft // 2 + 1, -1)
    solutions = find_sliding_window_params(transform, in_spec, out_spec)
    params = solutions[0]

    test_stream_equivalent(
        transform,
        SlidingWindowStream(transform, params, in_spec, out_spec),
        in_data=in_spec.new_randn_arrays(200),
    )
    return solutions, params


def _solver_worker(
    config: Tuple[int, bool, str, Optional[int]],
    log_queue: mp.Queue,
) -> None:
    handler = _QueueLogHandler(log_queue)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    n_fft, center, pad_mode, hop_length = config
    logger.info("Starting solver for config %s", config)
    run_solver(n_fft, center, pad_mode, hop_length)
    logger.info("Solver finished successfully.")


def _init_solver_state() -> None:
    if "solver_process" in st.session_state:
        return
    st.session_state["solver_process"] = None
    st.session_state["solver_config"] = None
    st.session_state["solver_status"] = "idle"
    st.session_state["solver_error"] = None
    st.session_state["solver_log_queue"] = None
    st.session_state["solver_logs"] = []


def _drain_log_queue() -> None:
    log_queue = st.session_state.get("solver_log_queue")
    if not log_queue:
        return
    logs = st.session_state.setdefault("solver_logs", [])
    while True:
        try:
            kind, payload = log_queue.get_nowait()
        except queue.Empty:
            break
        if kind == "log":
            logs.append(payload)


def _cleanup_log_queue() -> None:
    log_queue = st.session_state.get("solver_log_queue")
    if not log_queue:
        return
    _drain_log_queue()
    try:
        log_queue.close()
    except Exception:
        pass
    try:
        log_queue.join_thread()
    except Exception:
        pass
    st.session_state["solver_log_queue"] = None


def _stop_solver_process() -> None:
    proc: Optional[mp.Process] = st.session_state.get("solver_process")  # type: ignore[assignment]
    if not proc:
        _cleanup_log_queue()
        return
    if proc.is_alive():
        proc.terminate()
    proc.join(timeout=5)
    st.session_state["solver_process"] = None
    _cleanup_log_queue()


def _poll_solver_process() -> None:
    proc: Optional[mp.Process] = st.session_state.get("solver_process")  # type: ignore[assignment]
    if not proc:
        return
    if proc.is_alive():
        return

    exit_code = proc.exitcode
    st.session_state["solver_process"] = None
    if exit_code == 0:
        st.session_state["solver_status"] = "done"
        st.session_state["solver_error"] = None
    else:
        st.session_state["solver_status"] = "error"
        st.session_state["solver_error"] = f"Solver process exited with code {exit_code}"
    _cleanup_log_queue()


def _start_solver_process(config: Tuple[int, bool, str, Optional[int]]) -> None:
    _stop_solver_process()
    log_queue: mp.Queue = mp.Queue()
    st.session_state["solver_log_queue"] = log_queue
    st.session_state["solver_logs"] = []

    proc = mp.Process(target=_solver_worker, args=(config, log_queue), daemon=True)
    proc.start()
    st.session_state["solver_process"] = proc
    st.session_state["solver_config"] = config
    st.session_state["solver_status"] = "running"
    st.session_state["solver_error"] = None


def app() -> None:
    st.set_page_config(page_title="Melspectrogram Sliding Window", layout="wide")
    _init_solver_state()
    _poll_solver_process()
    _drain_log_queue()

    st.title("Mel Spectrogram Sliding-Window Solver")
    st.write(
        "Use the controls below to explore how different spectrogram settings impact "
        "the sliding-window parameters inferred by `torchstream`."
    )

    code_col, ui_col, status_col = st.columns([1, 1, 1], gap="medium", border=True)

    with code_col:
        st.subheader("Source code")
        st.code(inspect.getsource(run_solver), language="python")

    with ui_col:
        n_fft = st.slider(
            "n_fft",
            min_value=16,
            max_value=400,
            value=20,
            step=2,
            help="Window size used by torchaudio.transforms.Spectrogram.",
        )
        center = st.toggle(
            "center",
            value=True,
            help="When on, the audio tensor is padded so that the t-th frame is centered.",
        )
        pad_mode = st.radio(
            "pad_mode",
            options=("reflect", "constant", "replicate"),
            index=0,
            horizontal=True,
            help="Padding mode that torchaudio uses when center=True.",
        )
        hop_length = st.slider(
            "hop_length",
            min_value=0,
            max_value=n_fft,
            value=0,
            help="Stride between windows.",
        )
        hop_length_value: Optional[int] = None if hop_length == 0 else hop_length

        config_signature = (n_fft, center, pad_mode, hop_length_value)
        if st.session_state.get("solver_config") != config_signature:
            _start_solver_process(config_signature)

        st.caption("Changes kick off a background solver in a separate process.")

    with status_col:
        st.subheader("Solver status")
        status = st.session_state.get("solver_status", "idle")
        error = st.session_state.get("solver_error")
        proc: Optional[mp.Process] = st.session_state.get("solver_process")  # type: ignore[assignment]

        if status == "running" and proc and proc.is_alive():
            st.info("Solver running in background. Adjust sliders freely.")
        elif status == "done":
            st.success("Solver finished successfully.")
        elif status == "error":
            st.error(error or "Solver failed. Check logs for details.")
        else:
            st.write("Adjust any control to start the solver.")

        st.write("---")
        st.caption("Solver logs (latest first)")
        logs = st.session_state.get("solver_logs", [])
        if logs:
            st.code("\n".join(logs[-50:]), language="text")
        else:
            st.caption("No logs yet.")

        if status == "running":
            st.caption("Refreshing status...")
            time.sleep(0.5)
            st.rerun()


if __name__ == "__main__":
    mp.freeze_support()  # Needed for compatibility on Windows executables.
    app()
