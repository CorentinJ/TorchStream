import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

from opentelemetry import trace

from dev_tools.tracing import log_tracing_profile

tracer = trace.get_tracer(__name__)

# Add a logging filter that injects repo-relative paths for clickable links
_REPO_ROOT = Path(__file__).resolve().parent


# Ensure every LogRecord carries a repo-relative path attribute early
_orig_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    record = _orig_factory(*args, **kwargs)
    try:
        rel = Path(record.pathname).resolve().relative_to(_REPO_ROOT)
        record.relpath = rel.as_posix()
    except Exception:
        record.relpath = getattr(record, "pathname", getattr(record, "filename", "<unknown>"))
    return record


logging.setLogRecordFactory(_record_factory)

_tracing_session_ctx = log_tracing_profile("pytest-session")


def pytest_sessionstart(session):
    _tracing_session_ctx.__enter__()


def pytest_sessionfinish(session, exitstatus):
    _tracing_session_ctx.__exit__(None, None, None)


def pytest_runtest_setup(item):
    span_name = f"test:{item.originalname}"
    ctx = tracer.start_as_current_span(span_name)
    span = ctx.__enter__()
    item._otel_test_span_ctx = ctx
    item._otel_test_span = span
    try:
        span.set_attribute("pytest.nodeid", item.nodeid)
        span.set_attribute("pytest.filepath", str(item.fspath))
    except Exception:
        pass


def pytest_runtest_makereport(item, call):
    span = getattr(item, "_otel_test_span", None)
    if span is None or call.when != "call":
        return
    try:
        if call.excinfo is not None:
            # record exception on the span
            span.record_exception(call.excinfo.value)
            span.set_attribute("pytest.outcome", "failed")
        else:
            span.set_attribute("pytest.outcome", "passed")
    except Exception:
        pass


def pytest_runtest_teardown(item, nextitem):
    ctx = getattr(item, "_otel_test_span_ctx", None)
    if ctx is None:
        return
    try:
        ctx.__exit__(None, None, None)
    except Exception:
        pass
    finally:
        for attr in ("_otel_test_span_ctx", "_otel_test_span"):
            try:
                delattr(item, attr)
            except Exception:
                pass


def _beep_ok():
    if platform.system() == "Windows":
        try:
            import winsound

            # pleasant short double-beep
            winsound.Beep(880, 180)
            winsound.Beep(660, 180)
            return
        except Exception:
            pass
    elif platform.system() == "Darwin":  # macOS
        try:
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass
    else:  # Linux/*nix fallback: terminal bell
        pass
    # universal fallback (may be muted in many terminals/OSes)
    sys.stdout.write("\a")
    sys.stdout.flush()


def _beep_fail():
    return
    if platform.system() == "Windows":
        try:
            import winsound

            # lower, longer tones
            winsound.Beep(220, 300)
            winsound.Beep(196, 500)
            return
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Basso.aiff"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass
    else:
        pass
    sys.stdout.write("\a\a")
    sys.stdout.flush()


def pytest_terminal_summary(terminalreporter, exitstatus):
    """
    Called after all tests have run and results are known.
    Skips beeping if no tests actually ran (e.g. during discovery or collection-only).
    """
    # Only beep if at least one test was executed (not just collected)
    if not hasattr(terminalreporter, "stats"):
        return

    # Count actual test outcomes (passed, failed, skipped, etc.)
    test_outcomes = sum(
        len(terminalreporter.stats.get(key, []))
        for key in ("passed", "failed", "skipped", "error", "xfailed", "xpassed")
    )
    if test_outcomes == 0:
        return  # No tests actually ran

    # Get test duration in seconds
    duration = getattr(terminalreporter, "_sessionstarttime", None)
    if duration is not None:
        duration = time.time() - duration
        print(f"\nTests duration: {duration:.2f} seconds")

    if duration is None or duration > 10.0:
        if exitstatus == 0:
            _beep_ok()
        else:
            _beep_fail()
