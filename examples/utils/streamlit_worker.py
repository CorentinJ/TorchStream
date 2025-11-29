from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


class _QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            pass


@dataclass
class _RunState:
    run_id: Optional[str] = None
    thread: Optional[threading.Thread] = None
    log_queue: Optional["queue.Queue[str]"] = None
    log_handler: Optional[logging.Handler] = None
    logs: list[str] = field(default_factory=list)
    result_box: dict[str, Any] = field(default_factory=dict)
    done_event: threading.Event = field(default_factory=threading.Event)
    callback_done: bool = False


def run_managed_thread(
    func: Callable[..., Any],
    run_id: str,
    log_container,
    on_complete: Callable[[Any], None],
    *,
    state_key: str = "managed_thread",
    func_args: Tuple[Any, ...] = (),
    func_kwargs: Optional[dict[str, Any]] = None,
    log_height: int = 300,
):
    if func_kwargs is None:
        func_kwargs = {}

    if state_key not in st.session_state:
        st.session_state[state_key] = _RunState()
    s: _RunState = st.session_state[state_key]

    if s.thread is not None and s.thread.is_alive() and s.run_id != run_id:
        setattr(s.thread, "streamlit_script_run_ctx", None)
        s.logs.append(f"[manager] Requested interrupt for previous run {s.run_id!r}")

        s.thread.join()
        s.logs.append(f"[manager] Previous run {s.run_id!r} has stopped")

        s.thread = None
        s.done_event = threading.Event()
        s.result_box = {}
        s.callback_done = False

    if (s.thread is None or not s.thread.is_alive()) and s.run_id != run_id:
        s.log_queue = queue.Queue()
        s.logs = []
        s.done_event = threading.Event()
        s.result_box = {}
        s.callback_done = False
        s.run_id = run_id

        target_logger = logging.getLogger()
        if s.log_handler is not None:
            try:
                target_logger.removeHandler(s.log_handler)
            except ValueError:
                pass

        s.log_handler = _QueueLogHandler(s.log_queue)
        s.log_handler.setLevel(logging.INFO)
        target_logger.addHandler(s.log_handler)
        target_logger.setLevel(min(target_logger.level, logging.INFO))

        def worker():
            try:
                result = func(*func_args, **func_kwargs)
                s.result_box["status"] = "ok"
                s.result_box["value"] = result
            except Exception as exc:
                s.result_box["status"] = "error"
                s.result_box["value"] = exc
            finally:
                s.done_event.set()

        s.thread = threading.Thread(target=worker, daemon=True)
        # Enables new thread to modify UI elements from current thread
        add_script_run_ctx(s.thread, get_script_run_ctx())
        s.thread.start()
        s.logs.append(f"[manager] Started run {run_id!r} (thread={s.thread.name})")

    if s.log_queue is not None:
        while True:
            try:
                msg = s.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                s.logs.append(msg)

    log_text = "\n".join(s.logs)
    log_container.text_area(
        "Logs",
        value=log_text,
        height=log_height,
        key=f"{state_key}_text_area",
    )

    if s.thread is not None and not s.thread.is_alive() and not s.callback_done and s.done_event.is_set():
        target_logger = logging.getLogger()
        if s.log_handler is not None:
            try:
                target_logger.removeHandler(s.log_handler)
            except ValueError:
                pass

        status = s.result_box.get("status")
        result = s.result_box.get("value", None)

        if status == "ok":
            s.logs.append("[manager] Worker completed successfully")
        elif status == "error":
            s.logs.append(f"[manager] Worker raised: {result!r}")
        else:
            s.logs.append("[manager] Worker finished with no result")

        def _callback_runner():
            on_complete(result)

        threading.Thread(target=_callback_runner, daemon=True).start()
        s.callback_done = True
