"""Main window: queue table, settings, DnD import, review wall, open actions."""

from __future__ import annotations

import logging
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
    DEFAULT_BATCH_SIZE,
    BATCH_SIZE_CHOICES,
)
from framesentry.core.input_discovery import DiscoveryResult, discover_videos, normalize_video_key
from framesentry.core.types import VideoJob, VideoStatus
from framesentry.core.ffmpeg import resolve_ffmpeg
from framesentry.scanner.preprocess import default_intermediate_dir
from framesentry.scanner.worker import ScanSettings
from framesentry.ui.ort_probe import OrtProbeController
from framesentry.ui.review_panel import MODE_ALL_HITS, ReviewPanel
from framesentry.ui.scan_controller import ScanWorker

logger = logging.getLogger(__name__)

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
        self._ort_probe: OrtProbeController | None = None

        self._build_ui()
        # ORT probe is started AFTER show() from main.py — never block __init__.
        self.ort_label.setText("ORT: CUDA Provider：检测中…")

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
        self.btn_intermediate = QPushButton("中间MP4目录…")
        self.btn_start = QPushButton("开始扫描")
        self.btn_cancel_cur = QPushButton("取消当前")
        self.btn_cancel_q = QPushButton("取消队列")
        for b in (
            self.btn_add_files,
            self.btn_add_folder,
            self.btn_clear,
            self.btn_output,
            self.btn_intermediate,
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
        self.btn_intermediate.clicked.connect(self._on_choose_intermediate)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_cancel_cur.clicked.connect(self._on_cancel_current)
        self.btn_cancel_q.clicked.connect(self._on_cancel_queue)

        # Review / open actions
        open_bar = QHBoxLayout()
        self.btn_open_review = QPushButton("打开审核页")
        self.btn_open_folder = QPushButton("打开结果文件夹")
        self.btn_locate_image = QPushButton("定位当前图片")
        self.btn_open_image = QPushButton("打开原图")
        self.btn_open_logs = QPushButton("打开日志目录")
        for b in (
            self.btn_open_review,
            self.btn_open_folder,
            self.btn_locate_image,
            self.btn_open_image,
            self.btn_open_logs,
        ):
            open_bar.addWidget(b)
        open_bar.addStretch(1)
        root.addLayout(open_bar)

        self.btn_open_review.clicked.connect(self._on_open_review_page)
        self.btn_open_folder.clicked.connect(self._on_open_result_folder)
        self.btn_locate_image.clicked.connect(self._on_locate_current_image)
        self.btn_open_image.clicked.connect(self._on_open_current_image)
        self.btn_open_logs.clicked.connect(self._on_open_logs)

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

        settings.addWidget(QLabel("批大小:"))
        self.batch_combo = QComboBox()
        for b in BATCH_SIZE_CHOICES:
            self.batch_combo.addItem(str(b), b)
        self.batch_combo.setCurrentIndex(
            list(BATCH_SIZE_CHOICES).index(DEFAULT_BATCH_SIZE)
            if DEFAULT_BATCH_SIZE in BATCH_SIZE_CHOICES else 0
        )
        settings.addWidget(self.batch_combo)

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

    # --- ORT background probe --------------------------------------------

    def start_ort_probe(self) -> None:
        """Start non-blocking ORT provider probe (call after window.show())."""
        self.ort_label.setText("ORT: CUDA Provider：检测中…")
        self._ort_probe = OrtProbeController(self)
        self._ort_probe.result.connect(self._on_ort_probe_result)
        self._ort_probe.error.connect(self._on_ort_probe_error)
        self._ort_probe.status.connect(self._on_ort_probe_status)
        self._ort_probe.start()

    def _on_ort_probe_status(self, msg: str) -> None:
        if "检测中" in msg or msg.startswith("ORT"):
            # Keep compact status in the label while probing
            if "检测中" in msg:
                self.ort_label.setText("ORT: CUDA Provider：检测中…")

    def _on_ort_probe_result(self, info: object) -> None:
        try:
            available = list(getattr(info, "available_providers", ()) or ())
            cuda_listed = bool(getattr(info, "cuda_listed", False))
            providers = ", ".join(available) or "(none / ort not installed)"
            cuda_ok = "是" if cuda_listed else "否"
            self.ort_label.setText(
                f"ORT providers: [{providers}] | 检测到 CUDA Provider：{cuda_ok}"
            )
            if not cuda_listed:
                self.radio_gpu.setEnabled(False)
                self.radio_gpu.setToolTip(
                    "当前 ORT 未列出 CUDAExecutionProvider（不等于 GPU session 已成功；"
                    "真正的 CUDA session 在扫描创建检测器时验证，失败会 FAILED）"
                )
                self.radio_cpu.setChecked(True)
            else:
                self.radio_gpu.setEnabled(True)
                self.radio_gpu.setToolTip("")
        except Exception as exc:  # noqa: BLE001
            logger.exception("ORT probe result apply failed")
            self._gui_error("ORT 信息更新失败", str(exc))

    def _on_ort_probe_error(self, message: str) -> None:
        self.ort_label.setText("ORT: CUDA Provider：失败")
        logger.error("ORT probe error: %s", message)
        self._log(f"ORT 检测失败: {message}")

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


    def _on_choose_intermediate(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择中间 MP4 目录", self._intermediate_dir)
        if d:
            self._intermediate_dir = d
            self.statusBar().showMessage(f"中间MP4目录: {self._intermediate_dir}")


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
        status_text = job.status.value
        if job.status == VideoStatus.PREPROCESSING:
            status_text = "PREPROCESSING (快速转 MP4)"
        elif job.status == VideoStatus.SCANNING:
            status_text = "SCANNING (GPU 扫描)"
        elif job.status == VideoStatus.READY:
            status_text = "READY"
        self.table.item(idx, 2).setText(status_text)
        if job.progress < 0:
            self.table.item(idx, 3).setText("…")
        else:
            self.table.item(idx, 3).setText(f"{job.progress:.0f}%")
        self.table.item(idx, 4).setText(str(job.hit_count))
        self.table.item(idx, 5).setText(job.error)

    def _selected_job(self) -> VideoJob | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if 0 <= idx < len(self._jobs):
            return self._jobs[idx]
        return None

    def _on_table_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._jobs):
            job = self._jobs[idx]
            if job.output_dir and Path(job.output_dir, "results.json").is_file():
                try:
                    self.review.set_mode(MODE_ALL_HITS)
                    self.review.load_review(job.output_dir)
                except Exception as exc:  # noqa: BLE001
                    self._log(f"加载复核失败: {exc}")
                    logger.exception("load review failed")

    # --- Open actions ----------------------------------------------------

    def _gui_error(self, title: str, message: str) -> None:
        logger.error("%s: %s", title, message)
        try:
            QMessageBox.warning(self, title, message)
        except Exception:  # noqa: BLE001
            pass
        self._log(f"{title}: {message}")

    def _on_open_review_page(self) -> None:
        try:
            from framesentry.core.platform_open import PlatformOpenError, open_html_in_browser

            review_dir = self.review.review_dir()
            if review_dir is None:
                job = self._selected_job()
                if job and job.output_dir:
                    review_dir = Path(job.output_dir)
            if review_dir is None:
                self._gui_error("打开审核页", "没有可用的复核目录（请先完成扫描或选择任务）")
                return
            index = Path(review_dir) / "index.html"
            if not index.is_file():
                self._gui_error("打开审核页", f"未找到 index.html: {index}")
                return
            open_html_in_browser(index)
        except Exception as exc:  # noqa: BLE001
            self._gui_error("打开审核页失败", str(exc))

    def _on_open_result_folder(self) -> None:
        try:
            from framesentry.core.platform_open import open_directory

            review_dir = self.review.review_dir()
            if review_dir is None:
                job = self._selected_job()
                if job and job.output_dir:
                    review_dir = Path(job.output_dir)
            if review_dir is None or not Path(review_dir).exists():
                self._gui_error("打开结果文件夹", "没有可用的结果文件夹")
                return
            open_directory(review_dir)
        except Exception as exc:  # noqa: BLE001
            self._gui_error("打开结果文件夹失败", str(exc))

    def _on_locate_current_image(self) -> None:
        try:
            from framesentry.core.platform_open import reveal_in_file_manager

            path = self.review.current_frame_path()
            if not path:
                self._gui_error("定位当前图片", "请先在审核墙中选择一张缩略图")
                return
            reveal_in_file_manager(path)
        except Exception as exc:  # noqa: BLE001
            self._gui_error("定位当前图片失败", str(exc))

    def _on_open_current_image(self) -> None:
        try:
            from framesentry.core.platform_open import open_path_with_default_app

            path = self.review.current_frame_path()
            if not path:
                self._gui_error("打开原图", "请先在审核墙中选择一张缩略图")
                return
            open_path_with_default_app(path)
        except Exception as exc:  # noqa: BLE001
            self._gui_error("打开原图失败", str(exc))

    def _on_open_logs(self) -> None:
        try:
            from framesentry.core.platform_open import open_logs_directory

            opened = open_logs_directory()
            self._log(f"已打开日志目录: {opened}")
        except Exception as exc:  # noqa: BLE001
            self._gui_error("打开日志目录失败", str(exc))

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
        batch_data = self.batch_combo.currentData()
        batch_size = int(batch_data) if batch_data is not None else DEFAULT_BATCH_SIZE
        if device == "cpu":
            batch_size = 1
        return ScanSettings(
            sample_fps=fps,
            threshold=float(self.threshold_spin.value()),
            device=device,
            output_root=self._output_root,
            batch_size=batch_size,
            intermediate_dir=self._intermediate_dir,
        )

    def _on_start(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "FrameSentry", "扫描已在进行中")
            return
        paths = [j.path for j in self._jobs if j.status == VideoStatus.WAITING]
        if not paths:
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
            from framesentry.detectors.nudenet_backend import create_nudenet_backend

            return create_nudenet_backend(device=device)

        if not resolve_ffmpeg():
            QMessageBox.warning(
                self,
                "FrameSentry",
                "未找到 FFmpeg。\n\n请设置环境变量 FRAMESENTRY_FFMPEG 指向本地 ffmpeg，"
                "或将其加入 PATH。\nFrameSentry 不会自动下载 FFmpeg；未找到时不会启动 NudeNet 扫描。",
            )
            self._log("错误: 未找到 FFmpeg，已中止启动")
            return

        Path(settings.intermediate_dir).mkdir(parents=True, exist_ok=True)

        self._worker = ScanWorker(self)
        self._worker.configure(paths, settings, factory)
        self._worker.job_started.connect(self._on_job_started)
        self._worker.job_status.connect(self._on_job_status)
        self._worker.job_progress.connect(self._on_job_progress)
        self._worker.job_finished.connect(self._on_job_finished)
        self._worker.queue_finished.connect(self._on_queue_finished)
        self._worker.finished.connect(self._on_worker_thread_finished)
        self._worker.log_message.connect(self._log)
        self.btn_start.setEnabled(False)
        self._worker.start()
        self._log(
            f"开始扫描 {len(paths)} 个视频 (device={device}, batch={settings.batch_size}, "
            f"intermediate={settings.intermediate_dir})"
        )

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
            # Status refined by job_status (PREPROCESSING vs SCANNING)
            if self._jobs[idx].status == VideoStatus.WAITING:
                self._jobs[idx].status = VideoStatus.PREPROCESSING
            self._jobs[idx].progress = 0.0
            self._jobs[idx].error = ""
            self._update_row(idx)

    def _on_job_status(self, path: str, status_name: str) -> None:
        idx = self._find_row(path)
        if idx < 0:
            return
        try:
            status = VideoStatus(status_name)
        except ValueError:
            return
        self._jobs[idx].status = status
        self._update_row(idx)
        # Distinct log labels: 快速转 MP4 vs GPU 扫描
        if status == VideoStatus.PREPROCESSING:
            self._log(f"快速转 MP4: {path}")
        elif status == VideoStatus.SCANNING:
            self._log(f"GPU/CPU 扫描: {path}")

    def _on_job_progress(self, path: str, pct: float, message: str) -> None:
        idx = self._find_row(path)
        if idx >= 0:
            # pct < 0 → indeterminate (unknown frame_count); keep message in error/status col lightly
            self._jobs[idx].progress = pct
            if pct < 0 and message:
                # Show sampled/timecode/hits text in status bar; table shows "…"
                self.statusBar().showMessage(message, 2000)
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
            # Auto-switch to this task's all-hits review wall immediately
            try:
                self.table.selectRow(idx)
                self.review.set_mode(MODE_ALL_HITS)
                self.review.load_from_memory(
                    video_path=path,
                    review_dir=job.output_dir,
                    hits=result.get("hits") or [],
                    events=result.get("events") or [],
                    meta=result.get("meta"),
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"复核面板更新失败: {exc}")
                logger.exception("review panel update failed")
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

    def _on_worker_thread_finished(self) -> None:
        """QThread finished → deleteLater → clear reference (no worker accumulation)."""
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

    # --- feedback --------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_pane.append(msg)

    def _toast(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 8000)
