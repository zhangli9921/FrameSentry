"""Cross-platform helpers to open files / folders / reveal in file manager."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class PlatformOpenError(RuntimeError):
    """Raised when a platform open/reveal action fails."""


def _abs_path(path: str | Path) -> Path:
    p = Path(path)
    try:
        return p.resolve()
    except OSError:
        return p.absolute()


def open_path_with_default_app(path: str | Path) -> None:
    """Open a file or directory with the OS default application."""
    abs_p = _abs_path(path)
    if not abs_p.exists():
        raise PlatformOpenError(f"path does not exist: {abs_p}")
    try:
        if sys.platform == "win32":
            os.startfile(str(abs_p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(abs_p)], check=False)
        else:
            subprocess.run(["xdg-open", str(abs_p)], check=False)
    except OSError as exc:
        logger.exception("open_path_with_default_app failed: %s", abs_p)
        raise PlatformOpenError(f"failed to open {abs_p}: {exc}") from exc


def open_directory(path: str | Path) -> None:
    """Open a directory in the system file manager."""
    abs_p = _abs_path(path)
    if abs_p.is_file():
        abs_p = abs_p.parent
    if not abs_p.is_dir():
        raise PlatformOpenError(f"directory does not exist: {abs_p}")
    open_path_with_default_app(abs_p)


def reveal_in_file_manager(path: str | Path) -> None:
    """Select/reveal a file in the OS file manager.

    Windows: ``explorer /select,<abs_path>`` (single argv; handles spaces/Unicode).
    Others: best-effort open parent directory.
    """
    abs_p = _abs_path(path)
    if not abs_p.exists():
        raise PlatformOpenError(f"path does not exist: {abs_p}")

    if sys.platform == "win32":
        # Pass "/select,<path>" as one argument so spaces/Chinese paths work.
        # Do not use shell=True.
        arg = f"/select,{abs_p}"
        try:
            subprocess.run(["explorer", arg], check=False)
        except OSError as exc:
            logger.exception("explorer /select failed: %s", abs_p)
            raise PlatformOpenError(f"failed to reveal {abs_p}: {exc}") from exc
        return

    # Non-Windows: open parent folder (no reliable select across DEs).
    parent = abs_p.parent if abs_p.is_file() else abs_p
    try:
        open_path_with_default_app(parent)
    except PlatformOpenError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PlatformOpenError(
            f"reveal not supported on this platform; could not open parent: {exc}"
        ) from exc


def open_html_in_browser(index_html: str | Path) -> None:
    """Open a local HTML file via Qt desktop services when available, else default app."""
    abs_p = _abs_path(index_html)
    if not abs_p.is_file():
        raise PlatformOpenError(f"HTML file does not exist: {abs_p}")
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(abs_p)))
        if not ok:
            raise PlatformOpenError(f"QDesktopServices.openUrl returned False for {abs_p}")
        return
    except ImportError:
        pass
    except PlatformOpenError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("QDesktopServices open failed, falling back: %s", exc)
    open_path_with_default_app(abs_p)


def logs_directory() -> Path:
    """Return the FrameSentry logs directory (same as logging_setup.default_log_dir)."""
    from framesentry.core.logging_setup import default_log_dir

    return default_log_dir()


def open_logs_directory() -> Path:
    """Ensure and open the logs directory. Returns the path opened."""
    log_dir = logs_directory()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PlatformOpenError(f"cannot create log dir {log_dir}: {exc}") from exc
    open_directory(log_dir)
    return log_dir
