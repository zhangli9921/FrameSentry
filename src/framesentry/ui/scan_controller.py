"""QThread-based scan queue — one video failure must not stop the queue."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QMutex, QThread, Signal, QObject

from framesentry.core.types import VideoStatus
from framesentry.scanner.worker import ScanCancelled, ScanSettings, scan_video


class ScanWorker(QThread):
    """Background worker processing a queue of video paths."""

    job_started = Signal(str)  # path
    job_progress = Signal(str, float, str)  # path, pct, message
    job_finished = Signal(str, str, object)  # path, status_name, result_or_error
    queue_finished = Signal()
    log_message = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mutex = QMutex()
        self._queue: list[str] = []
        self._settings: ScanSettings | None = None
        self._cancel_current = False
        self._cancel_queue = False
        self._detector_factory: Any = None

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

    def request_cancel_queue(self) -> None:
        self._mutex.lock()
        try:
            self._cancel_queue = True
            self._cancel_current = True
        finally:
            self._mutex.unlock()

    def _flags(self) -> tuple[bool, bool]:
        self._mutex.lock()
        try:
            return self._cancel_current, self._cancel_queue
        finally:
            self._mutex.unlock()

    def run(self) -> None:
        assert self._settings is not None
        assert self._detector_factory is not None
        try:
            detector = self._detector_factory()
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"Detector init failed: {exc}")
            # Mark all as failed
            self._mutex.lock()
            paths = list(self._queue)
            self._queue.clear()
            self._mutex.unlock()
            for p in paths:
                self.job_finished.emit(p, VideoStatus.FAILED.value, str(exc))
            self.queue_finished.emit()
            return

        while True:
            cancel_cur, cancel_q = self._flags()
            if cancel_q:
                self._mutex.lock()
                remaining = list(self._queue)
                self._queue.clear()
                self._mutex.unlock()
                for p in remaining:
                    self.job_finished.emit(p, VideoStatus.CANCELLED.value, "queue cancelled")
                break

            self._mutex.lock()
            if not self._queue:
                self._mutex.unlock()
                break
            path = self._queue.pop(0)
            self._cancel_current = False
            self._mutex.unlock()

            self.job_started.emit(path)
            self.log_message.emit(f"Scanning: {path}")

            def on_progress(pct: float, msg: str, _p=path) -> None:
                self.job_progress.emit(_p, pct, msg)

            def should_cancel() -> bool:
                c, q = self._flags()
                return c or q

            try:
                result = scan_video(
                    path,
                    detector,
                    self._settings,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                )
                self.job_finished.emit(path, VideoStatus.COMPLETED.value, result)
                self.log_message.emit(
                    f"Completed: {path} hits={result.get('hit_count', 0)}"
                )
            except ScanCancelled:
                self.job_finished.emit(path, VideoStatus.CANCELLED.value, "cancelled")
                self.log_message.emit(f"Cancelled: {path}")
            except Exception as exc:  # noqa: BLE001 — one failure must not stop queue
                self.job_finished.emit(path, VideoStatus.FAILED.value, str(exc))
                self.log_message.emit(f"Failed: {path} — {exc}")

        self.queue_finished.emit()
