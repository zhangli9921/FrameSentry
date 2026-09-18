"""Application entry point with ordered startup diagnostics."""

from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
from pathlib import Path

# Module-level flag for startup watchdog (set when main window is shown).
_main_window_shown = False
_watchdog_cancelled = False
_log = logging.getLogger("framesentry.startup")


def _install_excepthooks() -> None:
    """Route uncaught exceptions to the file log."""

    def _hook(exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        try:
            _log.error(
                "Uncaught exception",
                exc_info=(exc_type, exc, tb),
            )
        except Exception:  # noqa: BLE001
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook

    if hasattr(threading, "excepthook"):

        def _thread_hook(args) -> None:  # type: ignore[no-untyped-def]
            try:
                _log.error(
                    "Uncaught thread exception thread=%s",
                    getattr(args, "thread", None),
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )
            except Exception:  # noqa: BLE001
                pass

        threading.excepthook = _thread_hook  # type: ignore[assignment]


def _dump_all_thread_stacks(reason: str = "startup watchdog") -> None:
    """Dump every thread stack via sys._current_frames() into the log once."""
    frames = sys._current_frames()
    lines = [f"=== {reason}: dumping {len(frames)} thread stack(s) ==="]
    for thread_id, frame in frames.items():
        lines.append(f"--- thread id={thread_id} ---")
        lines.extend(traceback.format_stack(frame))
    _log.error("\n".join(lines))


def dump_all_thread_stacks(reason: str = "startup watchdog") -> None:
    """Public alias for tests."""
    _dump_all_thread_stacks(reason)


def _start_startup_watchdog(timeout_sec: float = 12.0) -> threading.Timer:
    """Daemon timer: if main window not shown in time, dump stacks once."""

    def _fire() -> None:
        if _main_window_shown or _watchdog_cancelled:
            return
        _dump_all_thread_stacks("startup watchdog fired (main window not shown)")

    timer = threading.Timer(timeout_sec, _fire)
    timer.daemon = True
    timer.start()
    return timer


def start_startup_watchdog(timeout_sec: float = 12.0) -> threading.Timer:
    """Public alias for tests."""
    return _start_startup_watchdog(timeout_sec)


def cancel_startup_watchdog(timer: threading.Timer | None) -> None:
    global _watchdog_cancelled
    _watchdog_cancelled = True
    if timer is not None:
        try:
            timer.cancel()
        except Exception:  # noqa: BLE001
            pass


def mark_main_window_shown() -> None:
    global _main_window_shown
    _main_window_shown = True
    _log.info("main window shown")


def reset_startup_flags_for_tests() -> None:
    """Reset module flags between tests."""
    global _main_window_shown, _watchdog_cancelled
    _main_window_shown = False
    _watchdog_cancelled = False


def main() -> int:
    """Launch the FrameSentry GUI with diagnostics-first startup ordering."""
    # 1) stdlib + lightweight logging only — no PySide6/cv2/numpy/ort/nudenet/MainWindow
    from framesentry.core.logging_setup import setup_logging

    # 2) setup_logging first
    log_path = setup_logging()
    _log.info(
        "process start pid=%s argv=%s cwd=%s python=%s log_path=%s",
        os.getpid(),
        list(sys.argv),
        str(Path.cwd()),
        sys.version.split()[0],
        str(log_path) if log_path else "(none)",
    )

    # 3) excepthooks → file log
    _install_excepthooks()

    watchdog = _start_startup_watchdog(12.0)

    # 4) import PySide6
    _log.info("startup phase: importing PySide6")
    from PySide6.QtWidgets import QApplication

    _log.info("startup phase: PySide6 import done")

    # 5) QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("FrameSentry")
    app.setOrganizationName("FrameSentry")
    _log.info("QApplication created")

    # 6) import + construct MainWindow (lazy — not at module top)
    _log.info("startup phase: importing MainWindow")
    from framesentry.ui.main_window import MainWindow

    _log.info("startup phase: MainWindow import done")
    window = MainWindow()
    _log.info("MainWindow constructed")

    # 7) show, then mark + cancel watchdog
    window.show()
    mark_main_window_shown()
    cancel_startup_watchdog(watchdog)

    # 8) background ORT/provider probe (never blocks __init__/show)
    try:
        window.start_ort_probe()
    except Exception:  # noqa: BLE001
        _log.exception("failed to start ORT probe")

    try:
        code = int(app.exec())
    finally:
        _log.info("event loop exit")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
