from __future__ import annotations

import logging
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from examples.utils.streamlit_dynamic_logs import create_logbox, update_logs

_RUN_MANAGED_THREAD_ID_KEY = "streamlit_run_managed_thread_id"


class _StreamlitLogHandler(logging.Handler):
    def __init__(self, logs_id: str) -> None:
        super().__init__()
        self.logs_id = logs_id
        self._past_logs = []

    def emit(self, record: logging.LogRecord) -> None:
        if get_script_run_ctx(suppress_warning=True) is None:
            print("Received interrupt, stopping cleanly")
            sys.exit(0)

        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self._past_logs.append(msg)

        self.refresh()

    def refresh(self):
        update_logs(self.logs_id, self._past_logs)


@dataclass
class _RunState:
    run_id: Optional[str] = None
    thread: Optional[threading.Thread] = None


def run_managed_thread(
    func: Callable[..., Any],
    run_id: str,
    job_id: str,
    func_args: Tuple[Any, ...] = (),
    func_kwargs: Optional[dict[str, Any]] = None,
    log_height: int = 300,
):
    if func_kwargs is None:
        func_kwargs = {}

    # Unique session identifier added to run id to avoid collisions between different sessions
    if _RUN_MANAGED_THREAD_ID_KEY not in st.session_state:
        st.session_state[_RUN_MANAGED_THREAD_ID_KEY] = uuid.uuid4().hex
    session_id = st.session_state[_RUN_MANAGED_THREAD_ID_KEY]
    job_id = f"{job_id}__{session_id}"
    run_id = f"{run_id}__{session_id}"

    # Unique state for any run of this job
    if job_id not in st.session_state:
        st.session_state[job_id] = _RunState()
    state: _RunState = st.session_state[job_id]

    create_logbox(run_id, height=log_height)

    if state.thread is not None and state.thread.is_alive() and state.run_id != run_id:
        setattr(state.thread, "streamlit_script_run_ctx", None)
        state.thread.join()
        state.thread = None

    if (state.thread is None or not state.thread.is_alive()) and state.run_id != run_id:
        state.run_id = run_id

        target_logger = logging.getLogger()
        target_logger.handlers = [h for h in target_logger.handlers if not isinstance(h, _StreamlitLogHandler)]
        log_handler = _StreamlitLogHandler(run_id)
        log_handler.setLevel(logging.INFO)
        target_logger.addHandler(log_handler)
        target_logger.setLevel(min(target_logger.level, logging.INFO))

        def worker():
            try:
                func(*func_args, **func_kwargs)
            except Exception as exc:
                st.exception(exc)
                raise exc

        state.thread = threading.Thread(target=worker, daemon=True)
        # Enables new thread to modify UI elements from current thread
        add_script_run_ctx(state.thread, get_script_run_ctx())
        state.thread.start()
