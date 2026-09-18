"""FFmpeg discovery and stream-copy remux (no re-encode, no shell)."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from framesentry.core.logging_setup import get_logger

logger = get_logger(__name__)

CancelCheck = Callable[[], bool]

# Keep stderr tails bounded to avoid unbounded FFmpeg log growth.
_STDERR_MAX_LINES = 80
_STDERR_MAX_BYTES = 64 * 1024


class FFmpegNotFoundError(RuntimeError):
    """Raised when no usable ffmpeg binary can be resolved."""


class RemuxCancelled(Exception):
    """Raised when remux is cancelled by the user."""


class RemuxError(RuntimeError):
    """Raised when stream-copy remux fails (e.g. codec cannot mux to MP4)."""

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


@dataclass
class RemuxResult:
    source_path: str
    intermediate_path: str
    duration_sec: float
    exit_code: int
    ffmpeg_path: str
    stderr_tail: str = ""


@dataclass
class FFmpegProcessHandle:
    """Tracks the single active ffmpeg child for cancel/kill."""

    proc: subprocess.Popen[bytes] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, proc: subprocess.Popen[bytes] | None) -> None:
        with self.lock:
            self.proc = proc

    def get(self) -> subprocess.Popen[bytes] | None:
        with self.lock:
            return self.proc

    def kill_windows_safe(self) -> None:
        """Terminate the active child without shell=True."""
        with self.lock:
            proc = self.proc
            self.proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        logger.warning("ffmpeg child did not exit after kill")
        except OSError as exc:
            logger.warning("failed to kill ffmpeg: %s", exc)


def resolve_ffmpeg(
    *,
    env: dict[str, str] | None = None,
    nearby_roots: Sequence[str | Path] | None = None,
) -> str | None:
    """Resolve ffmpeg binary.

    Order: env ``FRAMESENTRY_FFMPEG`` → optional nearby paths → ``shutil.which("ffmpeg")``.
    Does not auto-download. Returns None if not found.
    """
    environ = env if env is not None else os.environ
    configured = (environ.get("FRAMESENTRY_FFMPEG") or "").strip()
    if configured:
        p = Path(configured)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
        # On Windows, X_OK is often true for any existing file; still accept is_file.
        if p.is_file():
            return str(p.resolve())
        logger.warning("FRAMESENTRY_FFMPEG set but not a file: %s", configured)

    roots: list[Path] = []
    if nearby_roots is not None:
        roots.extend(Path(r) for r in nearby_roots)
    else:
        # Optional nearby: cwd, package-adjacent tools/, repo tools/, common bin names.
        roots.append(Path.cwd())
        try:
            pkg = Path(__file__).resolve().parents[2]  # .../src or site-packages
            roots.append(pkg)
            roots.append(pkg.parent)
            roots.append(pkg / "tools")
            roots.append(pkg.parent / "tools")
        except (IndexError, OSError):
            pass

    candidates = (
        "ffmpeg",
        "ffmpeg.exe",
        Path("bin") / "ffmpeg",
        Path("bin") / "ffmpeg.exe",
        Path("tools") / "ffmpeg",
        Path("tools") / "ffmpeg.exe",
    )
    for root in roots:
        for name in candidates:
            cand = root / name if not isinstance(name, str) or os.sep in name or "/" in str(name) else root / name
            # normalize Path/str mix
            cand = Path(cand)
            if cand.is_file():
                return str(cand.resolve())

    which = shutil.which("ffmpeg")
    if which:
        return which
    return None


def build_remux_argv(ffmpeg_bin: str, src: str | Path, dst_part: str | Path) -> list[str]:
    """Stream-copy remux argv (no shell, no re-encode, video-only)."""
    return [
        str(ffmpeg_bin),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-fflags",
        "+genpts+discardcorrupt",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(dst_part),
    ]


def _drain_stderr(proc: subprocess.Popen[bytes], sink: list[bytes], stop: threading.Event) -> None:
    assert proc.stderr is not None
    total = 0
    try:
        while not stop.is_set():
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            if total < _STDERR_MAX_BYTES and len(sink) < _STDERR_MAX_LINES * 4:
                sink.append(chunk)
                total += len(chunk)
    except OSError:
        pass


def remux_stream_copy(
    src: str | Path,
    dst: str | Path,
    *,
    ffmpeg_bin: str | None = None,
    should_cancel: CancelCheck | None = None,
    process_handle: FFmpegProcessHandle | None = None,
) -> RemuxResult:
    """Remux ``src`` → ``dst`` via stream copy into a ``.part`` then atomic replace.

    On failure/cancel: deletes the ``.part`` file; never deletes ``src`` or a
    previously successful ``dst``.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.is_file():
        raise FileNotFoundError(f"source video not found: {src}")

    bin_path = ffmpeg_bin or resolve_ffmpeg()
    if not bin_path:
        raise FFmpegNotFoundError(
            "FFmpeg not found. Set FRAMESENTRY_FFMPEG or install ffmpeg on PATH."
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # Requirement: write DST.part.mp4 then atomic replace to DST.mp4
    part_path = dst_path.with_name(dst_path.stem + ".part.mp4")

    if part_path.exists():
        try:
            part_path.unlink()
        except OSError:
            pass

    argv = build_remux_argv(bin_path, src_path, part_path)
    logger.info("ffmpeg remux argv=%s", argv)

    t0 = time.perf_counter()
    stderr_chunks: list[bytes] = []
    stop_reader = threading.Event()
    creationflags = 0
    if os.name == "nt":
        # New process group so we can terminate without a shell; avoid console flash.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise RemuxError(f"failed to start ffmpeg: {exc}", exit_code=None) from exc

    if process_handle is not None:
        process_handle.set(proc)

    reader = threading.Thread(
        target=_drain_stderr, args=(proc, stderr_chunks, stop_reader), daemon=True
    )
    reader.start()

    cancelled = False
    try:
        while True:
            if should_cancel and should_cancel():
                cancelled = True
                if process_handle is not None:
                    process_handle.kill_windows_safe()
                else:
                    try:
                        proc.terminate()
                        proc.wait(timeout=2.0)
                    except Exception:  # noqa: BLE001
                        try:
                            proc.kill()
                        except OSError:
                            pass
                break
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.05)
        # Ensure process finished
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait(timeout=2.0)
    finally:
        stop_reader.set()
        reader.join(timeout=2.0)
        if process_handle is not None:
            process_handle.set(None)
        # Close stderr to unblock reader if needed
        try:
            if proc.stderr:
                proc.stderr.close()
        except OSError:
            pass

    duration = time.perf_counter() - t0
    stderr_bytes = b"".join(stderr_chunks)[-_STDERR_MAX_BYTES:]
    stderr_tail = stderr_bytes.decode("utf-8", errors="replace")[-4000:]
    exit_code = proc.returncode if proc.returncode is not None else -1

    if cancelled:
        _safe_unlink(part_path)
        raise RemuxCancelled(f"remux cancelled: {src_path}")

    if exit_code != 0 or not part_path.is_file() or part_path.stat().st_size <= 0:
        _safe_unlink(part_path)
        raise RemuxError(
            f"PREPROCESS FAILED: ffmpeg exit={exit_code} for {src_path}. "
            f"stderr_tail={stderr_tail[-500:]}",
            exit_code=exit_code,
            stderr_tail=stderr_tail,
        )

    try:
        os.replace(str(part_path), str(dst_path))
    except OSError as exc:
        _safe_unlink(part_path)
        raise RemuxError(
            f"atomic replace failed: {exc}",
            exit_code=exit_code,
            stderr_tail=stderr_tail,
        ) from exc

    return RemuxResult(
        source_path=str(src_path),
        intermediate_path=str(dst_path),
        duration_sec=round(duration, 3),
        exit_code=exit_code,
        ffmpeg_path=bin_path,
        stderr_tail=stderr_tail,
    )


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not delete %s: %s", path, exc)
