"""Queue orchestrator: at most one FFmpeg, at most one file prepared ahead."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from framesentry.core.ffmpeg import FFmpegNotFoundError, FFmpegProcessHandle, RemuxCancelled
from framesentry.core.logging_setup import get_logger
from framesentry.core.types import VideoStatus
from framesentry.scanner.preprocess import PreprocessFailed, PreprocessResult, preprocess_video
from framesentry.scanner.worker import (
    ScanCancelled,
    ScanSettings,
    UnreadableVideoError,
    scan_video,
)

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float, str], None]  # path, pct, message
StatusCallback = Callable[[str, str], None]  # path, status_name
FinishedCallback = Callable[[str, str, Any], None]  # path, status, result
LogCallback = Callable[[str], None]


@dataclass
class PipelineStats:
    active_ffmpeg: int = 0
    prepared_ahead: int = 0
    preprocess_submitted: int = 0
    preprocess_completed: int = 0


@dataclass
class RemuxPipeline:
    """Serial scan with optional next-file FFmpeg preprocess (max 1 ahead).

    Invariants (always true):
    - ``active_ffmpeg <= 1``
    - ``prepared_ahead <= 1`` (only N+1 may be prepared while scanning N)
    """

    intermediate_dir: str
    settings: ScanSettings
    detector: Any
    ffmpeg_bin: str
    should_cancel_current: Callable[[], bool]
    should_cancel_queue: Callable[[], bool]
    clear_cancel_current: Callable[[], None] | None = None
    on_status: StatusCallback | None = None
    on_progress: ProgressCallback | None = None
    on_finished: FinishedCallback | None = None
    on_log: LogCallback | None = None
    stats: PipelineStats = field(default_factory=PipelineStats)

    def __post_init__(self) -> None:
        self._ffmpeg_handle = FFmpegProcessHandle()
        self._prep_lock = threading.Lock()
        self._prepared: dict[str, PreprocessResult | BaseException] = {}
        self._prep_future: Future[PreprocessResult] | None = None
        self._prep_path: str | None = None
        # Explicit ownership: which path's ffmpeg is running, and its role.
        # role "current" = preprocess of the file we are awaiting/scanning;
        # role "ahead" = preprocess of N+1 while scanning N.
        self._ffmpeg_owner_path: str | None = None
        self._ffmpeg_role: str | None = None  # "current" | "ahead" | None
        self._scanning_path: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fs-ffmpeg")
        self._closed = False

    def close(self) -> None:
        self._closed = True
        self.kill_ffmpeg()
        fut = self._prep_future
        if fut is not None:
            fut.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def kill_ffmpeg(self) -> None:
        """Kill whatever ffmpeg is active (queue cancel / shutdown)."""
        self._ffmpeg_handle.kill_windows_safe()
        with self._prep_lock:
            self.stats.active_ffmpeg = 0
            self._ffmpeg_owner_path = None
            self._ffmpeg_role = None

    def cancel_current_ffmpeg(self) -> None:
        """Kill ffmpeg only if it belongs to the *current* file, never ahead.

        Ownership is tracked explicitly via ``_ffmpeg_role`` / ``_ffmpeg_owner_path``
        — we do not guess by path alone when deciding whether to kill.
        """
        with self._prep_lock:
            role = self._ffmpeg_role
            owner = self._ffmpeg_owner_path
        if role == "ahead":
            # Ahead preprocess of N+1 must survive cancel-current of N.
            self._log(
                f"cancel_current: leaving ahead ffmpeg running (owner={owner})"
            )
            return
        if role == "current":
            self._log(f"cancel_current: killing current ffmpeg (owner={owner})")
            self.kill_ffmpeg()
            if owner:
                self._cleanup_part_for(owner)
            return
        # No active ffmpeg (or already finished) — nothing to kill.

    def _cleanup_part_for(self, path: str) -> None:
        try:
            from framesentry.scanner.preprocess import (
                cleanup_part_file,
                intermediate_mp4_path,
            )

            cleanup_part_file(intermediate_mp4_path(path, self.intermediate_dir))
        except Exception as exc:  # noqa: BLE001
            self._log(f"part cleanup failed for {path}: {exc}")

    def _ack_cancel_current(self) -> None:
        """Reset cancel-current flag so the next file is not auto-cancelled."""
        if self.clear_cancel_current is not None:
            self.clear_cancel_current()

    def _log(self, msg: str) -> None:
        logger.info("%s", msg)
        if self.on_log:
            self.on_log(msg)

    def _emit_status(self, path: str, status: VideoStatus) -> None:
        if self.on_status:
            self.on_status(path, status.value)

    def _cancel_flags(self) -> tuple[bool, bool]:
        return bool(self.should_cancel_current()), bool(self.should_cancel_queue())

    def _run_preprocess(self, path: str, *, role: str) -> PreprocessResult:
        with self._prep_lock:
            self.stats.active_ffmpeg = 1
            self.stats.preprocess_submitted += 1
            self._ffmpeg_owner_path = path
            self._ffmpeg_role = role

        def _cancel() -> bool:
            c, q = self._cancel_flags()
            if q:
                return True
            # cancel-current must only abort the *current* file's ffmpeg
            if c and role == "current":
                return True
            return False

        try:
            result = preprocess_video(
                path,
                self.intermediate_dir,
                ffmpeg_bin=self.ffmpeg_bin,
                should_cancel=_cancel,
                process_handle=self._ffmpeg_handle,
                reuse_existing=False,
            )
            return result
        finally:
            with self._prep_lock:
                if self._ffmpeg_owner_path == path:
                    self._ffmpeg_owner_path = None
                    self._ffmpeg_role = None
                self.stats.active_ffmpeg = 0
                self.stats.preprocess_completed += 1

    def _submit_preprocess(self, path: str) -> None:
        """Submit preprocess for ``path`` only if no ffmpeg is active / ahead slot free."""
        with self._prep_lock:
            if self._prep_future is not None and not self._prep_future.done():
                return  # already one in flight
            if path in self._prepared:
                return
            if self.stats.prepared_ahead >= 1 and self._prep_path != path:
                return
            self._prep_path = path
            self.stats.prepared_ahead = 1
            self._prep_future = self._executor.submit(
                self._run_preprocess, path, role="ahead"
            )

    def _await_prepared(self, path: str) -> PreprocessResult:
        """Block until ``path`` is preprocessed (may already be done / in flight)."""
        with self._prep_lock:
            cached = self._prepared.get(path)
            fut = self._prep_future
            fut_path = self._prep_path

        if isinstance(cached, PreprocessResult):
            return cached
        if isinstance(cached, BaseException):
            raise cached

        if fut is not None and fut_path == path:
            # This path is now the current file — promote ownership so cancel-current
            # can kill its ffmpeg (no longer protected as "ahead").
            with self._prep_lock:
                if self._prep_path == path and self._ffmpeg_role == "ahead":
                    self._ffmpeg_role = "current"
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                with self._prep_lock:
                    self._prepared[path] = exc
                    if self._prep_path == path:
                        self._prep_future = None
                        self._prep_path = None
                        self.stats.prepared_ahead = 0
                raise
            with self._prep_lock:
                self._prepared[path] = result
                if self._prep_path == path:
                    self._prep_future = None
                    self._prep_path = None
                    self.stats.prepared_ahead = 0
            return result

        # Not started yet — run inline (still respects active_ffmpeg<=1)
        with self._prep_lock:
            if self._prep_future is not None and not self._prep_future.done():
                # Wait for whatever is in flight first (should be this path or we mis-ordered)
                pass
        self._emit_status(path, VideoStatus.PREPROCESSING)
        self._log(f"快速转 MP4: {path}")
        try:
            result = self._run_preprocess(path, role="current")
        except Exception as exc:  # noqa: BLE001
            with self._prep_lock:
                self._prepared[path] = exc
            raise
        with self._prep_lock:
            self._prepared[path] = result
            self.stats.prepared_ahead = 0
        return result

    def _cancel_remaining(self, remaining: list[str]) -> None:
        # Snapshot owner before kill so we can clean its part file.
        with self._prep_lock:
            owner = self._ffmpeg_owner_path
        self.kill_ffmpeg()
        if owner:
            self._cleanup_part_for(owner)
        for p in remaining:
            self._cleanup_part_for(p)
            self._emit_status(p, VideoStatus.CANCELLED)
            if self.on_finished:
                self.on_finished(p, VideoStatus.CANCELLED.value, "queue cancelled")

    def run(self, paths: list[str]) -> None:
        """Process queue with scan(N) ∥ preprocess(N+1) only."""
        n = len(paths)
        i = 0
        try:
            while i < n:
                cancel_cur, cancel_q = self._cancel_flags()
                if cancel_q:
                    self._cancel_remaining(paths[i:])
                    break

                path = paths[i]
                next_path = paths[i + 1] if i + 1 < n else None

                # Ensure current is ready; while waiting, do not start N+2.
                try:
                    # If not already in flight as prepared-ahead, mark preprocessing.
                    with self._prep_lock:
                        already = path in self._prepared or (
                            self._prep_path == path and self._prep_future is not None
                        )
                    if not already:
                        self._emit_status(path, VideoStatus.PREPROCESSING)
                        self._log(f"快速转 MP4: {path}")
                    prep = self._await_prepared(path)
                except RemuxCancelled:
                    self._cleanup_part_for(path)
                    self._emit_status(path, VideoStatus.CANCELLED)
                    if self.on_finished:
                        self.on_finished(path, VideoStatus.CANCELLED.value, "cancelled")
                    self._log(f"CANCELLED (preprocess): {path}")
                    if self.should_cancel_queue():
                        self._cancel_remaining(paths[i + 1 :])
                        break
                    # cancel-current only: clear flag so B is not auto-cancelled
                    self._ack_cancel_current()
                    i += 1
                    continue
                except (PreprocessFailed, FFmpegNotFoundError, OSError) as exc:
                    self._emit_status(path, VideoStatus.FAILED)
                    if self.on_finished:
                        self.on_finished(path, VideoStatus.FAILED.value, str(exc))
                    self._log(f"PREPROCESS FAILED: {path} — {exc}")
                    i += 1
                    # Start next preprocess early if possible
                    if next_path and not self.should_cancel_queue():
                        self._emit_status(next_path, VideoStatus.PREPROCESSING)
                        self._submit_preprocess(next_path)
                    continue

                self._emit_status(path, VideoStatus.READY)

                # Submit ONLY next file preprocess while we scan current.
                if next_path is not None:
                    cancel_cur, cancel_q = self._cancel_flags()
                    if not cancel_q:
                        self._emit_status(next_path, VideoStatus.PREPROCESSING)
                        self._log(f"快速转 MP4 (ahead): {next_path}")
                        self._submit_preprocess(next_path)

                # Scan current from intermediate MP4 only.
                self._scanning_path = path
                self._emit_status(path, VideoStatus.SCANNING)
                self._log(f"GPU 扫描: {path}")

                def on_progress(pct: float, msg: str, _p=path) -> None:
                    if self.on_progress:
                        self.on_progress(_p, pct, msg)

                def should_cancel() -> bool:
                    c, q = self._cancel_flags()
                    return c or q

                try:
                    result = scan_video(
                        path,
                        self.detector,
                        self.settings,
                        scan_input_path=prep.intermediate_path,
                        preprocess_meta={
                            "source_video_path": prep.source_path,
                            "scan_input_path": prep.intermediate_path,
                            "intermediate_path": prep.intermediate_path,
                            "preprocess_duration_sec": prep.duration_sec,
                            "ffmpeg_exit_code": prep.exit_code,
                        },
                        on_progress=on_progress,
                        should_cancel=should_cancel,
                    )
                    if self.on_finished:
                        self.on_finished(path, VideoStatus.COMPLETED.value, result)
                    self._log(f"Completed: {path} hits={result.get('hit_count', 0)}")
                except ScanCancelled:
                    if self.should_cancel_queue():
                        # Queue cancel: kill current + ahead ffmpeg, cancel rest
                        self.kill_ffmpeg()
                        if self.on_finished:
                            self.on_finished(path, VideoStatus.CANCELLED.value, "cancelled")
                        self._log(f"CANCELLED: {path}")
                        remaining = paths[i + 1 :]
                        self._cancel_remaining(remaining)
                        break
                    # cancel-current only: do NOT kill ahead ffmpeg
                    if self.on_finished:
                        self.on_finished(path, VideoStatus.CANCELLED.value, "cancelled")
                    self._log(f"CANCELLED (scan only): {path}")
                    self._ack_cancel_current()
                except UnreadableVideoError as exc:
                    if self.on_finished:
                        self.on_finished(path, VideoStatus.FAILED.value, str(exc))
                    self._log(f"FAILED (unreadable): {path} — {exc}")
                except Exception as exc:  # noqa: BLE001 — one failure must not stop queue
                    if self.on_finished:
                        self.on_finished(path, VideoStatus.FAILED.value, str(exc))
                    self._log(f"FAILED: {path} — {exc}")

                self._scanning_path = None
                i += 1
        finally:
            self.close()
