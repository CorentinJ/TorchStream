from __future__ import annotations

import logging
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

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
    saved_logs: Optional[List[str]] = None


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

    # Session state (initialized once)
    if _RUN_MANAGED_THREAD_ID_KEY not in st.session_state:
        st.session_state[_RUN_MANAGED_THREAD_ID_KEY] = dict(
            session_id=uuid.uuid4().hex,
            thread_lock=threading.Lock(),
            active_run_id=None,
            run_history=dict(),
        )
    session_id = st.session_state[_RUN_MANAGED_THREAD_ID_KEY]["session_id"]
    thread_lock = st.session_state[_RUN_MANAGED_THREAD_ID_KEY]["thread_lock"]
    run_history = st.session_state[_RUN_MANAGED_THREAD_ID_KEY]["run_history"]

    job_id = f"{job_id}_{session_id}"
    run_id = f"{run_id}_{job_id}"

    # Check if we have results for this run_id, otherwise create state
    if run_id not in run_history:
        run_history[run_id] = _RunState()
    state: _RunState = run_history[run_id]

    create_logbox(run_id, height=log_height)

    # If the current run is this one, we can return early
    active_run_id = st.session_state[_RUN_MANAGED_THREAD_ID_KEY]["active_run_id"]
    if active_run_id == run_id:
        return

    # If the result is cached we can skip running the worker
    if state.saved_logs is not None:
        update_logs(run_id, state.saved_logs)
        return

    # Another run from the same job is active? We'll terminate it
    if active_run_id and active_run_id != run_id and active_run_id.endswith(job_id):
        other_state = run_history[active_run_id]
        setattr(other_state.thread, "streamlit_script_run_ctx", None)
        other_state.thread.join()

    # In any case now, we wait for our turn to run
    print(f"\x1b[32m{'LOCK ACQUIRE'}\x1b[39m")
    thread_lock.acquire()
    print(f"\x1b[32m{'LOCK ACQUIRED'}\x1b[39m")
    st.session_state[_RUN_MANAGED_THREAD_ID_KEY]["active_run_id"] = run_id

    target_logger = logging.getLogger()
    target_logger.handlers = [h for h in target_logger.handlers if not isinstance(h, _StreamlitLogHandler)]
    log_handler = _StreamlitLogHandler(run_id)
    log_handler.setLevel(logging.INFO)
    target_logger.addHandler(log_handler)
    target_logger.setLevel(min(target_logger.level, logging.INFO))

    def worker(thread_lock, state: _RunState, log_handler):
        try:
            func(*func_args, **func_kwargs)
        except Exception as exc:
            st.exception(exc)
            raise exc
        finally:
            print(f"\x1b[31m{'LOCK RELEASE'}\x1b[39m")
            thread_lock.release()
            state.saved_logs = list(log_handler._past_logs)

    state.thread = threading.Thread(target=worker, daemon=True, args=(thread_lock, state, log_handler))
    # Enables new thread to modify UI elements from current thread
    add_script_run_ctx(state.thread, get_script_run_ctx())
    state.thread.start()


def await_running_thread():
    thread_lock = st.session_state[_RUN_MANAGED_THREAD_ID_KEY]["thread_lock"]
    print(f"\x1b[32m{'AWAIT LOCK ACQUIRE'}\x1b[39m")
    thread_lock.acquire()
    thread_lock.release()
    print(f"\x1b[32m{'AWAIT LOCK ACQUIRED'}\x1b[39m")
