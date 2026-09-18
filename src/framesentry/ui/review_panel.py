"""Native review panel for DetectionEvent clusters (no HTML required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from framesentry.core.types import DetectionEvent, DetectionHit
from framesentry.storage.results import load_results

# Thumbnail side length — keep memory light; full image only on demand.
_THUMB_SIZE = 96
_THUMB_ROLE_PATH = Qt.ItemDataRole.UserRole
_THUMB_ROLE_EVENT_IDX = Qt.ItemDataRole.UserRole + 1


class ReviewPanel(QWidget):
    """Shows event clusters with lightweight thumbnails; expand raw hits; enlarge."""

    frame_selected = Signal(str)  # absolute frame path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._review_dir: Path | None = None
        self._hits: list[DetectionHit] = []
        self._events: list[DetectionEvent] = []
        self._video_path: str = ""

        self.event_list = QListWidget()
        self.event_list.setViewMode(QListView.ViewMode.IconMode)
        self.event_list.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self.event_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.event_list.setMovement(QListView.Movement.Static)
        self.event_list.setWordWrap(True)
        self.event_list.setSpacing(8)
        self.event_list.setGridSize(QSize(_THUMB_SIZE + 48, _THUMB_SIZE + 72))
        self.event_list.setUniformItemSizes(False)

        self.hit_list = QListWidget()
        self.preview = QLabel("选择事件以预览最佳帧")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(200)
        self.preview.setStyleSheet("background:#222; color:#ccc;")
        self.meta = QTextEdit()
        self.meta.setReadOnly(True)
        self.meta.setMaximumHeight(100)

        left = QVBoxLayout()
        left.addWidget(QLabel("Detection Events (缩略图)"))
        left.addWidget(self.event_list)
        left.addWidget(QLabel("Raw hits (event)"))
        left.addWidget(self.hit_list)

        left_w = QWidget()
        left_w.setLayout(left)

        right = QVBoxLayout()
        right.addWidget(self.preview, stretch=1)
        right.addWidget(self.meta)
        right_w = QWidget()
        right_w.setLayout(right)

        split = QSplitter()
        split.addWidget(left_w)
        split.addWidget(right_w)
        split.setStretchFactor(1, 2)

        layout = QHBoxLayout(self)
        layout.addWidget(split)

        self.event_list.currentRowChanged.connect(self._on_event_selected)
        self.hit_list.currentRowChanged.connect(self._on_hit_selected)
        self.event_list.itemDoubleClicked.connect(self._enlarge_best)
        self.hit_list.itemDoubleClicked.connect(self._enlarge_hit)

    def clear(self) -> None:
        self._review_dir = None
        self._hits = []
        self._events = []
        self._video_path = ""
        self.event_list.clear()
        self.hit_list.clear()
        self.preview.setPixmap(QPixmap())
        self.preview.setText("选择事件以预览最佳帧")
        self.meta.clear()

    def _set_meta(self) -> None:
        self.meta.setPlainText(
            "源视频: "
            + self._video_path
            + chr(10)
            + "原始命中: "
            + str(len(self._hits))
            + " | 事件: "
            + str(len(self._events))
            + chr(10)
            + "输出: "
            + str(self._review_dir)
        )

    def load_review(self, review_dir: str | Path) -> None:
        data = load_results(review_dir)
        self._review_dir = Path(review_dir)
        self._hits = data.get("_hits") or []
        self._events = data.get("_events") or []
        self._video_path = str(data.get("video_path") or "")
        self._populate_events()
        self._set_meta()

    def load_from_memory(
        self,
        *,
        video_path: str,
        review_dir: str | Path,
        hits: list[DetectionHit],
        events: list[DetectionEvent],
    ) -> None:
        self._review_dir = Path(review_dir)
        self._hits = list(hits)
        self._events = list(events)
        self._video_path = video_path
        self._populate_events()
        self._set_meta()

    def _populate_events(self) -> None:
        self.event_list.clear()
        self.hit_list.clear()
        for i, ev in enumerate(self._events):
            classes = ",".join(ev.classes) if ev.classes else "?"
            text = (
                f"{ev.start_str} → {ev.end_str}\n"
                f"{classes}\nmax={ev.max_score:.2f} n={ev.count}"
            )
            item = QListWidgetItem(text)
            thumb = self._load_thumbnail(ev.best_frame_filename)
            if thumb is not None:
                item.setIcon(QIcon(thumb))
            item.setData(_THUMB_ROLE_EVENT_IDX, i)
            item.setData(_THUMB_ROLE_PATH, ev.best_frame_filename)
            item.setToolTip(
                f"[{ev.start_str} → {ev.end_str}] {classes} "
                f"max={ev.max_score:.2f} hits={ev.count}"
            )
            self.event_list.addItem(item)

    def _load_thumbnail(self, filename: str | None) -> QPixmap | None:
        """Load a size-capped thumbnail without keeping full-resolution images."""
        path = self._frame_path(filename)
        if path is None:
            return None
        # QPixmap loads via Qt; we immediately scale down and discard full size.
        pix = QPixmap(str(path))
        if pix.isNull():
            return None
        return pix.scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _on_event_selected(self, row: int) -> None:
        self.hit_list.clear()
        if row < 0 or row >= len(self._events):
            return
        ev = self._events[row]
        for idx in ev.hit_indices:
            if 0 <= idx < len(self._hits):
                h = self._hits[idx]
                self.hit_list.addItem(
                    QListWidgetItem(
                        f"#{h.frame_index} {h.timestamp_str} {h.class_name} {h.score:.2f}"
                    )
                )
        self._show_frame(ev.best_frame_filename)

    def _on_hit_selected(self, row: int) -> None:
        ev_row = self.event_list.currentRow()
        if ev_row < 0 or ev_row >= len(self._events):
            return
        ev = self._events[ev_row]
        if row < 0 or row >= len(ev.hit_indices):
            return
        idx = ev.hit_indices[row]
        if 0 <= idx < len(self._hits):
            self._show_frame(self._hits[idx].frame_filename)

    def _frame_path(self, filename: str | None) -> Path | None:
        if not filename or not self._review_dir:
            return None
        p = self._review_dir / "frames" / filename
        return p if p.is_file() else None

    def _show_frame(self, filename: str | None) -> None:
        path = self._frame_path(filename)
        if path is None:
            self.preview.setText("无帧图")
            return
        # Preview scales to widget; avoid retaining extra full-size copies beyond Qt cache.
        pix = QPixmap(str(path))
        if pix.isNull():
            self.preview.setText("无法加载帧图")
            return
        scaled = pix.scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)
        self.frame_selected.emit(str(path))

    def _enlarge_best(self, _item: Any = None) -> None:
        row = self.event_list.currentRow()
        if 0 <= row < len(self._events):
            self._show_enlarged(self._events[row].best_frame_filename)

    def _enlarge_hit(self, _item: Any = None) -> None:
        ev_row = self.event_list.currentRow()
        hit_row = self.hit_list.currentRow()
        if ev_row < 0 or hit_row < 0:
            return
        ev = self._events[ev_row]
        if hit_row < len(ev.hit_indices):
            idx = ev.hit_indices[hit_row]
            if 0 <= idx < len(self._hits):
                self._show_enlarged(self._hits[idx].frame_filename)

    def _show_enlarged(self, filename: str | None) -> None:
        from PySide6.QtWidgets import QDialog, QVBoxLayout as QV

        path = self._frame_path(filename)
        if path is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(path.name)
        lay = QV(dlg)
        lab = QLabel()
        pix = QPixmap(str(path))
        lab.setPixmap(pix)
        lab.setScaledContents(False)
        lay.addWidget(lab)
        dlg.resize(min(1200, pix.width() + 40), min(800, pix.height() + 40))
        dlg.exec()
