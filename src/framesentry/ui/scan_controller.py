"""QThread-based scan queue — remux pipeline + batched scan; one failure must not stop queue."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QMutex, QObject, QThread, Signal

from framesentry.core.ffmpeg import FFmpegNotFoundError, resolve_ffmpeg
from framesentry.core.types import VideoStatus
from framesentry.scanner.pipeline import RemuxPipeline
from framesentry.scanner.worker import ScanSettings


class ScanWorker(QThread):
    """Background worker: FFmpeg remux (≤1 ahead) then GPU/CPU batched scan."""

    job_started = Signal(str)  # path
    job_progress = Signal(str, float, str)  # path, pct, message
    job_finished = Signal(str, str, object)  # path, status_name, result_or_error
    queue_finished = Signal()
    log_message = Signal(str)
    job_status = Signal(str, str)  # path, status_name (PREPROCESSING/READY/SCANNING/…)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mutex = QMutex()
        self._queue: list[str] = []
        self._settings: ScanSettings | None = None
        self._cancel_current = False
        self._cancel_queue = False
        self._detector_factory: Any = None
        self._pipeline: RemuxPipeline | None = None

    def configure(
        self,
        paths: list[str],
        settings: ScanSettings,
        detector_factory: Any,
    ) -> None:
        self._mutex.lock()
        try:
            self._queue = list(paths)
            self._settings = settings
            self._detector_factory = detector_factory
            self._cancel_current = False
            self._cancel_queue = False
        finally:
            self._mutex.unlock()

    def request_cancel_current(self) -> None:
        self._mutex.lock()
        try:
            self._cancel_current = True
        finally:
            self._mutex.unlock()
        pipe = self._pipeline
        if pipe is not None:
            # Only kill ffmpeg that belongs to the current file, never ahead.
            pipe.cancel_current_ffmpeg()

    def clear_cancel_current(self) -> None:
        """Reset cancel-current so the next file is not auto-cancelled."""
        self._mutex.lock()
        try:
            self._cancel_current = False
        finally:
            self._mutex.unlock()

    def request_cancel_queue(self) -> None:
        self._mutex.lock()
        try:
            self._cancel_queue = True
            self._cancel_current = True
        finally:
            self._mutex.unlock()
        pipe = self._pipeline
        if pipe is not None:
            pipe.kill_ffmpeg()

    def _flags(self) -> tuple[bool, bool]:
        self._mutex.lock()
        try:
            return self._cancel_current, self._cancel_queue
        finally:
            self._mutex.unlock()

    def run(self) -> None:
        assert self._settings is not None
        assert self._detector_factory is not None

        self._mutex.lock()
        paths = list(self._queue)
        settings = self._settings
        self._mutex.unlock()

        ffmpeg_bin = resolve_ffmpeg()
        if not ffmpeg_bin:
            msg = (
                "未找到 FFmpeg。请设置环境变量 FRAMESENTRY_FFMPEG 指向本地 ffmpeg，"
                "或将其加入 PATH。FrameSentry 不会自动下载 FFmpeg。"
            )
            self.log_message.emit(msg)
            for p in paths:
                self.job_finished.emit(p, VideoStatus.FAILED.value, msg)
            self.queue_finished.emit()
            return

        try:
            detector = self._detector_factory()
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"Detector init failed: {exc}")
            for p in paths:
                self.job_finished.emit(p, VideoStatus.FAILED.value, str(exc))
            self.queue_finished.emit()
            return

        intermediate_dir = (settings.intermediate_dir or "").strip()
        if not intermediate_dir:
            from framesentry.scanner.preprocess import default_intermediate_dir

            intermediate_dir = str(default_intermediate_dir())

        started_emitted: set[str] = set()

        def on_status(path: str, status_name: str) -> None:
            if path not in started_emitted and status_name in (
                VideoStatus.PREPROCESSING.value,
                VideoStatus.SCANNING.value,
                VideoStatus.READY.value,
            ):
                started_emitted.add(path)
                self.job_started.emit(path)
            self.job_status.emit(path, status_name)
            if status_name == VideoStatus.PREPROCESSING.value:
                self.log_message.emit(f"快速转 MP4: {path}")
            elif status_name == VideoStatus.SCANNING.value:
                self.log_message.emit(f"GPU/CPU 扫描: {path}")

        def on_progress(path: str, pct: float, message: str) -> None:
            self.job_progress.emit(path, pct, message)

        def on_finished(path: str, status_name: str, result: Any) -> None:
            self.job_finished.emit(path, status_name, result)
            if status_name == VideoStatus.COMPLETED.value and isinstance(result, dict):
                self.log_message.emit(
                    f"Completed: {path} hits={result.get('hit_count', 0)}"
                )
            else:
                self.log_message.emit(f"{status_name}: {path} — {result}")

        def on_log(msg: str) -> None:
            self.log_message.emit(msg)

        pipeline = RemuxPipeline(
            intermediate_dir=intermediate_dir,
            settings=settings,
            detector=detector,
            ffmpeg_bin=ffmpeg_bin,
            should_cancel_current=lambda: self._flags()[0],
            should_cancel_queue=lambda: self._flags()[1],
            clear_cancel_current=self.clear_cancel_current,
            on_status=on_status,
            on_progress=on_progress,
            on_finished=on_finished,
            on_log=on_log,
        )
        self._pipeline = pipeline
        try:
            pipeline.run(paths)
        except FFmpegNotFoundError as exc:
            self.log_message.emit(str(exc))
            for p in paths:
                self.job_finished.emit(p, VideoStatus.FAILED.value, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"Pipeline error: {exc}")
        finally:
            try:
                pipeline.close()
            except Exception:  # noqa: BLE001
                pass
            self._pipeline = None
            self.queue_finished.emit()
