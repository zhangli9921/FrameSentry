"""Main window: queue table, settings, DnD import, review."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from framesentry.core.config import (
    DEFAULT_SAMPLE_FPS,
    DEFAULT_THRESHOLD,
    SAMPLE_FPS_PRESETS,
    VIDEO_EXTENSIONS,
)
from framesentry.core.input_discovery import DiscoveryResult, discover_videos, normalize_video_key
from framesentry.core.types import VideoJob, VideoStatus
from framesentry.detectors.nudenet_backend import create_nudenet_backend, get_ort_provider_info
from framesentry.scanner.worker import ScanSettings
from framesentry.ui.review_panel import ReviewPanel
from framesentry.ui.scan_controller import ScanWorker

_DROP_HINT = "释放以添加视频或文件夹"
_DROP_IDLE = "拖放视频文件或文件夹到此处（或使用上方按钮）"


class DropZoneLabel(QLabel):
    """Visual drop-target hint (parent window still owns DnD events)."""

    def __init__(self, text: str = _DROP_IDLE, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(36)
        self.set_idle()

    def set_idle(self) -> None:
        self.setText(_DROP_IDLE)
        self.setStyleSheet(
            "QLabel { border: 1px dashed #888; padding: 8px; background: #fafafa; color: #555; }"
        )

    def set_active(self) -> None:
        self.setText(_DROP_HINT)
        self.setStyleSheet(
            "QLabel { border: 2px solid #1976d2; padding: 8px; background: #e3f2fd; color: #0d47a1; }"
        )


class MainWindow(QMainWindow):
    """FrameSentry desktop scanner main window with native Qt DnD."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FrameSentry — 视频内容初筛")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self._jobs: list[VideoJob] = []
        self._path_keys: set[str] = set()
        self._worker: ScanWorker | None = None
        self._output_root: str = str(Path.home() / "FrameSentryOutput")

        self._build_ui()
        self._refresh_ort_info()

    # --- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Toolbar buttons
        bar = QHBoxLayout()
        self.btn_add_files = QPushButton("添加文件")
        self.btn_add_folder = QPushButton("添加文件夹")
        self.btn_clear = QPushButton("清空列表")
        self.btn_output = QPushButton("输出目录…")
        self.btn_start = QPushButton("开始扫描")
        self.btn_cancel_cur = QPushButton("取消当前")
        self.btn_cancel_q = QPushButton("取消队列")
        for b in (
            self.btn_add_files,
            self.btn_add_folder,
            self.btn_clear,
            self.btn_output,
            self.btn_start,
            self.btn_cancel_cur,
            self.btn_cancel_q,
        ):
            bar.addWidget(b)
        bar.addStretch(1)
        root.addLayout(bar)

        self.btn_add_files.clicked.connect(self._on_add_files)
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_output.clicked.connect(self._on_choose_output)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_cancel_cur.clicked.connect(self._on_cancel_current)
        self.btn_cancel_q.clicked.connect(self._on_cancel_queue)

        # Settings row
        settings = QHBoxLayout()
        settings.addWidget(QLabel("设备:"))
        self.radio_gpu = QRadioButton("GPU")
        self.radio_cpu = QRadioButton("CPU")
        self.radio_cpu.setChecked(True)
        self.device_group = QButtonGroup(self)
        self.device_group.addButton(self.radio_gpu)
        self.device_group.addButton(self.radio_cpu)
        settings.addWidget(self.radio_gpu)
        settings.addWidget(self.radio_cpu)

        settings.addWidget(QLabel("采样FPS:"))
        self.fps_combo = QComboBox()
        for p in SAMPLE_FPS_PRESETS:
            self.fps_combo.addItem(str(int(p) if p == int(p) else p), p)
        self.fps_combo.setCurrentIndex(list(SAMPLE_FPS_PRESETS).index(DEFAULT_SAMPLE_FPS))
        self.fps_combo.setEditable(True)
        settings.addWidget(self.fps_combo)

        settings.addWidget(QLabel("阈值:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(DEFAULT_THRESHOLD)
        settings.addWidget(self.threshold_spin)

        self.ort_label = QLabel("ORT: …")
        self.ort_label.setWordWrap(True)
        settings.addWidget(self.ort_label, stretch=1)
        root.addLayout(settings)

        self.drop_zone = DropZoneLabel()
        root.addWidget(self.drop_zone)

        # Table + review splitter
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["名称", "路径", "状态", "进度", "命中", "错误"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_select)

        self.review = ReviewPanel()
        self.log_pane = QTextEdit()
        self.log_pane.setReadOnly(True)
        self.log_pane.setMaximumHeight(140)

        mid = QSplitter(Qt.Orientation.Vertical)
        mid.addWidget(self.table)
        mid.addWidget(self.review)
        mid.addWidget(self.log_pane)
        mid.setStretchFactor(0, 2)
        mid.setStretchFactor(1, 3)
        root.addWidget(mid, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"输出目录: {self._output_root}")

    # --- DnD -------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_zone.set_active()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # noqa: N802
        self.drop_zone.set_idle()
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.drop_zone.set_idle()
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        paths = [p for p in paths if p]
        if paths:
            self._ingest_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    # --- Path ingest (shared by buttons + DnD) ---------------------------

    def _ingest_paths(self, paths: list[str]) -> DiscoveryResult:
        result = discover_videos(paths, existing=self._path_keys)
        for video in result.videos:
            self._path_keys.add(normalize_video_key(video))
            job = VideoJob(path=video, name=Path(video).name)
            self._jobs.append(job)
            self._append_row(job)
        self._toast(result.toast_message)
        self._log(result.toast_message)
        if result.errors:
            for err in result.errors[:5]:
                self._log(f"忽略: {err}")
        return result

    def _on_add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "添加视频文件",
            "",
            f"Videos ({exts});;All (*)",
        )
        if files:
            self._ingest_paths(files)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "添加文件夹")
        if folder:
            self._ingest_paths([folder])

    def _on_clear(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "FrameSentry", "扫描进行中，无法清空")
            return
        self._jobs.clear()
        self._path_keys.clear()
        self.table.setRowCount(0)
        self.review.clear()

    def _on_choose_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出根目录", self._output_root)
        if d:
            self._output_root = d
            self.statusBar().showMessage(f"输出目录: {self._output_root}")

    # --- Table helpers ---------------------------------------------------

    def _append_row(self, job: VideoJob) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(job.name))
        self.table.setItem(r, 1, QTableWidgetItem(job.path))
        self.table.setItem(r, 2, QTableWidgetItem(job.status.value))
        self.table.setItem(r, 3, QTableWidgetItem(f"{job.progress:.0f}%"))
        self.table.setItem(r, 4, QTableWidgetItem(str(job.hit_count)))
        self.table.setItem(r, 5, QTableWidgetItem(job.error))

    def _find_row(self, path: str) -> int:
        key = normalize_video_key(path)
        for i, job in enumerate(self._jobs):
            if normalize_video_key(job.path) == key:
                return i
        return -1

    def _update_row(self, idx: int) -> None:
        job = self._jobs[idx]
        self.table.item(idx, 2).setText(job.status.value)
        self.table.item(idx, 3).setText(f"{job.progress:.0f}%")
        self.table.item(idx, 4).setText(str(job.hit_count))
        self.table.item(idx, 5).setText(job.error)

    def _on_table_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._jobs):
            job = self._jobs[idx]
            if job.output_dir and Path(job.output_dir, "results.json").is_file():
                try:
                    self.review.load_review(job.output_dir)
                except Exception as exc:  # noqa: BLE001
                    self._log(f"加载复核失败: {exc}")

    # --- ORT / device ----------------------------------------------------

    def _refresh_ort_info(self) -> None:
        info = get_ort_provider_info()
        providers = ", ".join(info.available_providers) or "(none / ort not installed)"
        gpu_ok = "是" if info.gpu_mode_usable else "否"
        self.ort_label.setText(
            f"ORT providers: [{providers}] | GPU模式可用: {gpu_ok}"
        )
        if not info.gpu_mode_usable:
            self.radio_gpu.setEnabled(False)
            self.radio_gpu.setToolTip("当前 ORT 未列出 CUDAExecutionProvider")
            self.radio_cpu.setChecked(True)
        else:
            self.radio_gpu.setEnabled(True)
            self.radio_gpu.setToolTip("")

    # --- Scan control ----------------------------------------------------

    def _current_settings(self) -> ScanSettings:
        device = "gpu" if self.radio_gpu.isChecked() else "cpu"
        fps_data = self.fps_combo.currentData()
        if fps_data is None:
            try:
                fps = float(self.fps_combo.currentText())
            except ValueError:
                fps = DEFAULT_SAMPLE_FPS
        else:
            fps = float(fps_data)
        return ScanSettings(
            sample_fps=fps,
            threshold=float(self.threshold_spin.value()),
            device=device,
            output_root=self._output_root,
        )

    def _on_start(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "FrameSentry", "扫描已在进行中")
            return
        paths = [j.path for j in self._jobs if j.status == VideoStatus.WAITING]
        if not paths:
            # Reset non-running jobs to waiting if user wants restart of failed/cancelled
            for j in self._jobs:
                if j.status in (VideoStatus.FAILED, VideoStatus.CANCELLED, VideoStatus.COMPLETED):
                    j.status = VideoStatus.WAITING
                    j.progress = 0.0
                    j.error = ""
                    j.hit_count = 0
            for i in range(len(self._jobs)):
                self._update_row(i)
            paths = [j.path for j in self._jobs if j.status == VideoStatus.WAITING]
        if not paths:
            QMessageBox.information(self, "FrameSentry", "没有待扫描的视频")
            return

        settings = self._current_settings()
        Path(settings.output_root).mkdir(parents=True, exist_ok=True)

        device = settings.device

        def factory() -> object:
            return create_nudenet_backend(device=device)

        self._worker = ScanWorker(self)
        self._worker.configure(paths, settings, factory)
        self._worker.job_started.connect(self._on_job_started)
        self._worker.job_progress.connect(self._on_job_progress)
        self._worker.job_finished.connect(self._on_job_finished)
        self._worker.queue_finished.connect(self._on_queue_finished)
        self._worker.log_message.connect(self._log)
        self.btn_start.setEnabled(False)
        self._worker.start()
        self._log(f"开始扫描 {len(paths)} 个视频 (device={device})")

    def _on_cancel_current(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel_current()
            self._log("请求取消当前视频…")

    def _on_cancel_queue(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel_queue()
            self._log("请求取消整个队列…")

    def _on_job_started(self, path: str) -> None:
        idx = self._find_row(path)
        if idx >= 0:
            self._jobs[idx].status = VideoStatus.SCANNING
            self._jobs[idx].progress = 0.0
            self._jobs[idx].error = ""
            self._update_row(idx)

    def _on_job_progress(self, path: str, pct: float, message: str) -> None:
        idx = self._find_row(path)
        if idx >= 0:
            self._jobs[idx].progress = pct
            self._update_row(idx)

    def _on_job_finished(self, path: str, status_name: str, result: object) -> None:
        idx = self._find_row(path)
        if idx < 0:
            return
        job = self._jobs[idx]
        try:
            status = VideoStatus(status_name)
        except ValueError:
            status = VideoStatus.FAILED
        job.status = status
        if status == VideoStatus.COMPLETED and isinstance(result, dict):
            job.progress = 100.0
            job.hit_count = int(result.get("hit_count") or 0)
            job.output_dir = str(result.get("review_dir") or "")
            job.error = ""
            # Auto-load review for completed
            try:
                self.review.load_from_memory(
                    video_path=path,
                    review_dir=job.output_dir,
                    hits=result.get("hits") or [],
                    events=result.get("events") or [],
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"复核面板更新失败: {exc}")
        elif status == VideoStatus.FAILED:
            job.error = str(result)
            job.progress = 0.0
        elif status == VideoStatus.CANCELLED:
            job.error = str(result)
        self._update_row(idx)

    def _on_queue_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self._log("队列完成")
        self.statusBar().showMessage("队列完成", 5000)

    # --- feedback --------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_pane.append(msg)

    def _toast(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 8000)
