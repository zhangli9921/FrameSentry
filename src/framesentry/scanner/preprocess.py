"""Intermediate MP4 preprocess (FFmpeg stream-copy remux) for scan input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from framesentry.core.ffmpeg import (
    FFmpegNotFoundError,
    FFmpegProcessHandle,
    RemuxCancelled,
    RemuxError,
    RemuxResult,
    remux_stream_copy,
    resolve_ffmpeg,
)
from framesentry.core.logging_setup import get_logger
from framesentry.storage.paths import path_stable_id

logger = get_logger(__name__)

CancelCheck = Callable[[], bool]

DEFAULT_INTERMEDIATE_DIRNAME = "FrameSentryIntermediate"


@dataclass
class PreprocessResult:
    source_path: str
    intermediate_path: str
    duration_sec: float
    exit_code: int
    ffmpeg_path: str
    status: str = "READY"  # READY | FAILED | CANCELLED


class PreprocessFailed(RuntimeError):
    """Preprocess failed; queue should continue with next video."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int | None = None,
        stderr_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail


def default_intermediate_dir() -> Path:
    """``%USERPROFILE%\\FrameSentryIntermediate`` / ``~/FrameSentryIntermediate``."""
    return Path.home() / DEFAULT_INTERMEDIATE_DIRNAME


def ensure_intermediate_dir(intermediate_dir: str | Path) -> Path:
    """Create intermediate dir if missing. Raises OSError if not writable."""
    d = Path(intermediate_dir)
    d.mkdir(parents=True, exist_ok=True)
    probe = d / ".framesentry_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise OSError(f"intermediate directory not writable: {d} ({exc})") from exc
    return d


def intermediate_mp4_name(source_path: str | Path) -> str:
    """``<stem>.<sha10(original_source_path)>.mp4``."""
    stem = Path(source_path).stem
    sid = path_stable_id(source_path)
    return f"{stem}.{sid}.mp4"


def intermediate_mp4_path(source_path: str | Path, intermediate_dir: str | Path) -> Path:
    return Path(intermediate_dir) / intermediate_mp4_name(source_path)


def part_path_for(intermediate: str | Path) -> Path:
    p = Path(intermediate)
    return p.with_name(p.stem + ".part.mp4")


def cleanup_part_file(intermediate: str | Path) -> None:
    part = part_path_for(intermediate)
    try:
        part.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("cleanup part failed %s: %s", part, exc)


def preprocess_video(
    source_path: str | Path,
    intermediate_dir: str | Path,
    *,
    ffmpeg_bin: str | None = None,
    should_cancel: CancelCheck | None = None,
    process_handle: FFmpegProcessHandle | None = None,
    reuse_existing: bool = True,
) -> PreprocessResult:
    """Stream-copy remux ``source`` into the intermediate dir.

    Review / results identity must still use ``source_path`` (original).
    """
    src = Path(source_path)
    out_dir = ensure_intermediate_dir(intermediate_dir)
    dst = intermediate_mp4_path(src, out_dir)

    bin_path = ffmpeg_bin or resolve_ffmpeg()
    if not bin_path:
        raise FFmpegNotFoundError(
            "FFmpeg not found. Set FRAMESENTRY_FFMPEG to a local ffmpeg binary "
            "or install ffmpeg on PATH. FrameSentry does not auto-download FFmpeg."
        )

    if reuse_existing and dst.is_file() and dst.stat().st_size > 0:
        logger.info(
            "reuse intermediate source=%s intermediate=%s",
            src,
            dst,
        )
        return PreprocessResult(
            source_path=str(src),
            intermediate_path=str(dst),
            duration_sec=0.0,
            exit_code=0,
            ffmpeg_path=bin_path,
            status="READY",
        )

    cleanup_part_file(dst)
    try:
        remux: RemuxResult = remux_stream_copy(
            src,
            dst,
            ffmpeg_bin=bin_path,
            should_cancel=should_cancel,
            process_handle=process_handle,
        )
    except RemuxCancelled:
        cleanup_part_file(dst)
        raise
    except RemuxError as exc:
        cleanup_part_file(dst)
        raise PreprocessFailed(
            str(exc),
            exit_code=exc.exit_code,
            stderr_tail=exc.stderr_tail,
        ) from exc

    logger.info(
        "preprocess ok source=%s intermediate=%s duration=%.3fs exit=%s",
        src,
        dst,
        remux.duration_sec,
        remux.exit_code,
    )
    return PreprocessResult(
        source_path=str(src),
        intermediate_path=str(dst),
        duration_sec=remux.duration_sec,
        exit_code=remux.exit_code,
        ffmpeg_path=remux.ffmpeg_path,
        status="READY",
    )
