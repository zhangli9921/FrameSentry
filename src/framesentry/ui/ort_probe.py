"""Background ORT / CUDA provider probe (no NudeNet session)."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)


class OrtProbeWorker(QObject):
    """Runs get_ort_provider_info off the UI thread."""

    finished = Signal(object)  # OrtProviderInfo-like or dict
    failed = Signal(str)
    log_line = Signal(str)

    def run(self) -> None:
        self.log_line.emit("ORT probe begin")
        logger.info("ORT probe begin")
        try:
            # Lazy import — never at MainWindow module import time for heavy deps path
            from framesentry.detectors.nudenet_backend import get_ort_provider_info

            info = get_ort_provider_info()
            # Also log version if available
            ort_ver = "unavailable"
            try:
                import onnxruntime as ort

                ort_ver = getattr(ort, "__version__", "unknown")
            except Exception as exc:  # noqa: BLE001
                logger.info("ORT version probe error: %s", exc)
            logger.info(
                "ORT probe end version=%s providers=%s cuda_listed=%s",
                ort_ver,
                list(info.available_providers),
                info.cuda_listed,
            )
            self.log_line.emit(
                f"ORT probe end version={ort_ver} providers={list(info.available_providers)}"
            )
            self.finished.emit(info)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ORT probe failed")
            self.failed.emit(str(exc))


class OrtProbeController(QObject):
    """Owns a QThread + OrtProbeWorker; emits UI-friendly signals."""

    result = Signal(object)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: OrtProbeWorker | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self.status.emit("CUDA Provider：检测中…")
        self._thread = QThread(self)
        self._worker = OrtProbeWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.log_line.connect(self.status.emit)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_finished(self, info: Any) -> None:
        self.result.emit(info)

    def _on_failed(self, message: str) -> None:
        self.error.emit(message)
