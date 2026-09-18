"""Logging configuration — stderr + user log file."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


_ENV_LOG_DIR = "FRAMESENTRY_LOG_DIR"
_file_handler: logging.FileHandler | None = None


def default_log_dir() -> Path:
    """Platform log directory (overridable via FRAMESENTRY_LOG_DIR)."""
    override = os.environ.get(_ENV_LOG_DIR)
    if override:
        return Path(override)

    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "FrameSentry" / "logs"
        return Path.home() / "AppData" / "Local" / "FrameSentry" / "logs"

    # Linux / macOS — XDG-ish user data, with optional platformdirs if present.
    try:
        from platformdirs import user_log_dir  # type: ignore[import-not-found]

        return Path(user_log_dir("FrameSentry", appauthor=False))
    except ImportError:
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Logs" / "FrameSentry"
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
        return base / "FrameSentry" / "logs"


def setup_logging(level: int = logging.INFO) -> Path | None:
    """Configure stderr + timestamped file log under the user log dir.

    Returns the log file path when a file handler was added, else None.
    Safe to call more than once: level is updated; file handler is not duplicated.
    """
    global _file_handler

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    has_stderr = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr
        for h in root.handlers
    )
    if not has_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(fmt)
        root.addHandler(stderr_handler)

    log_path: Path | None = None
    if _file_handler is None and not any(
        isinstance(h, logging.FileHandler) for h in root.handlers
    ):
        try:
            log_dir = default_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"framesentry_{stamp}.log"
            _file_handler = logging.FileHandler(log_path, encoding="utf-8")
            _file_handler.setFormatter(fmt)
            root.addHandler(_file_handler)
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not create file log: %s", exc)
            log_path = None
    elif _file_handler is not None:
        log_path = Path(_file_handler.baseFilename)

    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_environment_banner(
    *,
    active_providers: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Write version / runtime / ORT banner (no images or model weights)."""
    log = get_logger("framesentry")
    try:
        from framesentry import __version__
    except Exception:  # noqa: BLE001
        __version__ = "unknown"

    nudenet_ver = "unavailable"
    try:
        import importlib.metadata as md

        nudenet_ver = md.version("nudenet")
    except Exception:  # noqa: BLE001
        pass

    ort_ver = "unavailable"
    providers: list[str] = []
    try:
        import onnxruntime as ort

        ort_ver = getattr(ort, "__version__", "unknown")
        providers = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        pass

    log.info(
        "FrameSentry %s | Python %s | OS %s | NudeNet %s | ORT %s | "
        "available_providers=%s | active_detector_providers=%s",
        __version__,
        sys.version.split()[0],
        f"{os.name}/{sys.platform}",
        nudenet_ver,
        ort_ver,
        providers,
        list(active_providers) if active_providers is not None else "(not yet)",
    )


def reset_logging_for_tests() -> None:
    """Remove handlers so tests can re-init with a temp log dir."""
    global _file_handler
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass
        root.removeHandler(h)
    _file_handler = None
