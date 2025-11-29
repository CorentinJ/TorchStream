from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from examples.utils.streamlit_dynamic_logs import create_logbox, update_logs


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
    log_handler: Optional[logging.Handler] = None
    logs: list[str] = field(default_factory=list)
    result_box: dict[str, Any] = field(default_factory=dict)
    done_event: threading.Event = field(default_factory=threading.Event)
    callback_done: bool = False


def run_managed_thread(
    func: Callable[..., Any],
    run_id: str,
    on_complete: Callable[[Any], None],
    state_key: str = "managed_thread",
    func_args: Tuple[Any, ...] = (),
    func_kwargs: Optional[dict[str, Any]] = None,
    log_height: int = 300,
):
    if func_kwargs is None:
        func_kwargs = {}

    if state_key not in st.session_state:
        st.session_state[state_key] = _RunState()
    state: _RunState = st.session_state[state_key]

    create_logbox(state_key, height=log_height)

    if state.thread is not None and state.thread.is_alive() and state.run_id != run_id:
        setattr(state.thread, "streamlit_script_run_ctx", None)
        state.logs.append(f"[manager] Requested interrupt for previous run {state.run_id!r}")

        state.thread.join()
        state.logs.append(f"[manager] Previous run {state.run_id!r} has stopped")

        state.thread = None
        state.done_event = threading.Event()
        state.result_box = {}
        state.callback_done = False

    if (state.thread is None or not state.thread.is_alive()) and state.run_id != run_id:
        state.logs = []
        state.done_event = threading.Event()
        state.result_box = {}
        state.callback_done = False
        state.run_id = run_id

        target_logger = logging.getLogger()
        if state.log_handler is not None:
            try:
                target_logger.removeHandler(state.log_handler)
            except ValueError:
                pass

        state.log_handler = _StreamlitLogHandler(state_key)
        state.log_handler.setLevel(logging.INFO)
        target_logger.addHandler(state.log_handler)
        target_logger.setLevel(min(target_logger.level, logging.INFO))

        def worker():
            try:
                result = func(*func_args, **func_kwargs)
                state.result_box["status"] = "ok"
                state.result_box["value"] = result
            except Exception as exc:
                state.result_box["status"] = "error"
                state.result_box["value"] = exc
            finally:
                state.done_event.set()

        state.thread = threading.Thread(target=worker, daemon=True)
        # Enables new thread to modify UI elements from current thread
        add_script_run_ctx(state.thread, get_script_run_ctx())
        state.thread.start()
        state.logs.append(f"[manager] Started run {run_id!r} (thread={state.thread.name})")

    if (
        state.thread is not None
        and not state.thread.is_alive()
        and not state.callback_done
        and state.done_event.is_set()
    ):
        target_logger = logging.getLogger()
        if state.log_handler is not None:
            try:
                target_logger.removeHandler(state.log_handler)
            except ValueError:
                pass

        status = state.result_box.get("status")
        result = state.result_box.get("value", None)

        if status == "ok":
            state.logs.append("[manager] Worker completed successfully")
        elif status == "error":
            state.logs.append(f"[manager] Worker raised: {result!r}")
        else:
            state.logs.append("[manager] Worker finished with no result")

        def _callback_runner():
            on_complete(result)

        threading.Thread(target=_callback_runner, daemon=True).start()
        state.callback_done = True
