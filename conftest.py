import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

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
