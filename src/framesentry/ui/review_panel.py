"""Native review wall: all-hits thumbnail grid (default) + event best-frame mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from framesentry.core.types import DetectionEvent, DetectionHit
from framesentry.storage.html_report import group_hits_by_frame
from framesentry.storage.results import load_results

# Thumbnail width band (~180–240px); height follows aspect ratio via grid size.
_THUMB_W = 200
_THUMB_H = 140
_ROLE_PATH = Qt.ItemDataRole.UserRole
_ROLE_PAYLOAD = Qt.ItemDataRole.UserRole + 1

MODE_ALL_HITS = "all_hits"
MODE_EVENTS = "events"


class ReviewPanel(QWidget):
    """Scrollable hit-frame wall with optional event-representative mode."""

    frame_selected = Signal(str)  # absolute frame path
    mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._review_dir: Path | None = None
        self._hits: list[DetectionHit] = []
        self._events: list[DetectionEvent] = []
        self._video_path: str = ""
        self._meta: dict[str, Any] = {}
        self._mode = MODE_ALL_HITS
        self._groups: list[dict[str, Any]] = []
        self._current_frame_path: str | None = None

        # Mode toggle
        mode_row = QHBoxLayout()
        self.radio_all = QRadioButton("全部命中")
        self.radio_events = QRadioButton("事件代表帧")
        self.radio_all.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.radio_all)
        self._mode_group.addButton(self.radio_events)
        mode_row.addWidget(self.radio_all)
        mode_row.addWidget(self.radio_events)
        mode_row.addStretch(1)
        self.radio_all.toggled.connect(self._on_mode_toggled)
        self.radio_events.toggled.connect(self._on_mode_toggled)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)

        self.empty_label = QLabel("未检测到目标违规帧")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color:#888; padding:24px; font-size:14px;")
        self.empty_label.hide()

        # Thumbnail grid (used for both modes)
        self.frame_list = QListWidget()
        self.frame_list.setViewMode(QListView.ViewMode.IconMode)
        self.frame_list.setIconSize(QSize(_THUMB_W, _THUMB_H))
        self.frame_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.frame_list.setMovement(QListView.Movement.Static)
        self.frame_list.setWordWrap(True)
        self.frame_list.setSpacing(10)
        self.frame_list.setGridSize(QSize(_THUMB_W + 28, _THUMB_H + 88))
        self.frame_list.setUniformItemSizes(False)
        self.frame_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        # Back-compat alias used by older tests / callers
        self.event_list = self.frame_list

        self.preview = QLabel("选择缩略图以预览")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(160)
        self.preview.setMaximumHeight(280)
        self.preview.setStyleSheet("background:#222; color:#ccc;")

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(120)
        self.detail.setPlaceholderText("选中帧的详情将显示在这里")

        # Deprecated large raw-hit list kept empty/hidden for API stability
        self.hit_list = QListWidget()
        self.hit_list.hide()

        left = QVBoxLayout()
        left.addLayout(mode_row)
        left.addWidget(self.summary)
        left.addWidget(self.empty_label)
        left.addWidget(self.frame_list, stretch=1)
        left_w = QWidget()
        left_w.setLayout(left)

        right = QVBoxLayout()
        right.addWidget(QLabel("预览"))
        right.addWidget(self.preview, stretch=1)
        right.addWidget(QLabel("详情"))
        right.addWidget(self.detail)
        right_w = QWidget()
        right_w.setLayout(right)

        split = QSplitter()
        split.addWidget(left_w)
        split.addWidget(right_w)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self.frame_list.currentRowChanged.connect(self._on_tile_selected)
        self.frame_list.itemDoubleClicked.connect(self._enlarge_current)

    # --- public API ------------------------------------------------------

    def clear(self) -> None:
        self._review_dir = None
        self._hits = []
        self._events = []
        self._video_path = ""
        self._meta = {}
        self._groups = []
        self._current_frame_path = None
        self.frame_list.clear()
        self.hit_list.clear()
        self.preview.setPixmap(QPixmap())
        self.preview.setText("选择缩略图以预览")
        self.detail.clear()
        self.summary.setText("")
        self.empty_label.hide()
        self.frame_list.show()

    def current_frame_path(self) -> str | None:
        return self._current_frame_path

    def review_dir(self) -> Path | None:
        return self._review_dir

    def current_mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode == MODE_EVENTS:
            self.radio_events.setChecked(True)
        else:
            self.radio_all.setChecked(True)

    def load_review(self, review_dir: str | Path) -> None:
        data = load_results(review_dir)
        self._review_dir = Path(review_dir)
        self._hits = data.get("_hits") or []
        self._events = data.get("_events") or []
        self._video_path = str(data.get("video_path") or "")
        self._meta = dict(data.get("meta") or {})
        self._rebuild()

    def load_from_memory(
        self,
        *,
        video_path: str,
        review_dir: str | Path,
        hits: list[DetectionHit],
        events: list[DetectionEvent],
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._review_dir = Path(review_dir)
        self._hits = list(hits)
        self._events = list(events)
        self._video_path = video_path
        self._meta = dict(meta or {})
        self._rebuild()

    # --- internals -------------------------------------------------------

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self._mode = MODE_ALL_HITS if self.radio_all.isChecked() else MODE_EVENTS
        self.mode_changed.emit(self._mode)
        self._populate_tiles()

    def _rebuild(self) -> None:
        self._groups = group_hits_by_frame(self._hits)
        self._set_summary()
        self._populate_tiles()

    def _set_summary(self) -> None:
        parts = [
            f"源视频: {self._video_path}",
            f"原始命中: {len(self._hits)} | 帧组: {len(self._groups)} | 事件: {len(self._events)}",
            f"输出: {self._review_dir}",
        ]
        self.summary.setText("\n".join(parts))

    def _populate_tiles(self) -> None:
        self.frame_list.clear()
        self._current_frame_path = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("选择缩略图以预览")
        self.detail.clear()

        if self._mode == MODE_EVENTS:
            items_src = self._event_tile_specs()
        else:
            items_src = self._all_hit_tile_specs()

        if not items_src:
            self.empty_label.show()
            self.frame_list.hide()
            self.empty_label.setText("未检测到目标违规帧")
            return

        self.empty_label.hide()
        self.frame_list.show()

        for spec in items_src:
            text = spec["text"]
            item = QListWidgetItem(text)
            thumb = self._load_thumbnail(spec.get("filename"))
            if thumb is not None:
                item.setIcon(QIcon(thumb))
            item.setData(_ROLE_PATH, spec.get("filename"))
            item.setData(_ROLE_PAYLOAD, spec)
            item.setToolTip(spec.get("tooltip") or text)
            self.frame_list.addItem(item)

        # Auto-select first tile for preview without hiding the grid
        if self.frame_list.count() > 0:
            self.frame_list.setCurrentRow(0)

    def _all_hit_tile_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for g in self._groups:
            labels = []
            for d in g.get("detections") or []:
                labels.append(f"{d.get('class_name')} {float(d.get('score') or 0):.2f}")
            label_line = " | ".join(labels) if labels else "?"
            text = f"{g.get('timestamp_str')}\n{label_line}\n#{g.get('frame_index')}"
            specs.append(
                {
                    "text": text,
                    "filename": g.get("frame_filename"),
                    "tooltip": text.replace("\n", " "),
                    "kind": "hit_group",
                    "group": g,
                }
            )
        return specs

    def _event_tile_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for i, ev in enumerate(self._events):
            classes = ",".join(ev.classes) if ev.classes else "?"
            text = (
                f"{ev.start_str} → {ev.end_str}\n"
                f"{classes}\nmax={ev.max_score:.2f} n={ev.count}"
            )
            specs.append(
                {
                    "text": text,
                    "filename": ev.best_frame_filename,
                    "tooltip": (
                        f"事件{i + 1} [{ev.start_str} → {ev.end_str}] "
                        f"{classes} max={ev.max_score:.2f} hits={ev.count}"
                    ),
                    "kind": "event",
                    "event_index": i,
                    "event": ev,
                }
            )
        return specs

    def _load_thumbnail(self, filename: str | None) -> QPixmap | None:
        """Load a size-capped thumbnail; discard full-resolution copy after scale."""
        path = self._frame_path(filename)
        if path is None:
            return None
        pix = QPixmap(str(path))
        if pix.isNull():
            return None
        return pix.scaled(
            _THUMB_W,
            _THUMB_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _frame_path(self, filename: str | None) -> Path | None:
        if not filename or not self._review_dir:
            return None
        p = self._review_dir / "frames" / filename
        return p if p.is_file() else None

    def _on_tile_selected(self, row: int) -> None:
        if row < 0 or row >= self.frame_list.count():
            return
        item = self.frame_list.item(row)
        if item is None:
            return
        spec = item.data(_ROLE_PAYLOAD) or {}
        filename = item.data(_ROLE_PATH)
        self._show_frame(filename)
        self._show_detail(spec)

    def _show_detail(self, spec: dict[str, Any]) -> None:
        kind = spec.get("kind")
        lines: list[str] = []
        if kind == "hit_group":
            g = spec.get("group") or {}
            lines.append(f"时间: {g.get('timestamp_str')}")
            lines.append(f"帧索引: {g.get('frame_index')}")
            lines.append(f"文件: {g.get('frame_filename') or '(none)'}")
            for d in g.get("detections") or []:
                lines.append(
                    f"  - {d.get('class_name')}  score={float(d.get('score') or 0):.3f}"
                )
        elif kind == "event":
            ev: DetectionEvent | None = spec.get("event")
            if ev is not None:
                lines.append(f"事件: {ev.start_str} → {ev.end_str}")
                lines.append(f"类别: {', '.join(ev.classes)}")
                lines.append(f"最高分: {ev.max_score:.3f}  命中数: {ev.count}")
                lines.append(f"最佳帧: #{ev.best_frame_index} {ev.best_frame_filename}")
                # Compact hit summary (not a large permanent list widget)
                for idx in ev.hit_indices[:40]:
                    if 0 <= idx < len(self._hits):
                        h = self._hits[idx]
                        lines.append(
                            f"  hit#{idx} {h.timestamp_str} {h.class_name} {h.score:.2f}"
                        )
                if len(ev.hit_indices) > 40:
                    lines.append(f"  …共 {len(ev.hit_indices)} 条命中")
        self.detail.setPlainText("\n".join(lines))

    def _show_frame(self, filename: str | None) -> None:
        path = self._frame_path(filename)
        if path is None:
            self.preview.setText("无帧图")
            self._current_frame_path = None
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            self.preview.setText("无法加载帧图")
            self._current_frame_path = None
            return
        scaled = pix.scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)
        self._current_frame_path = str(path)
        self.frame_selected.emit(str(path))

    def _enlarge_current(self, _item: Any = None) -> None:
        from PySide6.QtWidgets import QDialog, QVBoxLayout as QV

        path = self._current_frame_path
        if not path:
            row = self.frame_list.currentRow()
            if row >= 0:
                item = self.frame_list.item(row)
                if item is not None:
                    p = self._frame_path(item.data(_ROLE_PATH))
                    path = str(p) if p else None
        if not path:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(Path(path).name)
        lay = QV(dlg)
        lab = QLabel()
        pix = QPixmap(path)
        lab.setPixmap(pix)
        lab.setScaledContents(False)
        lay.addWidget(lab)
        dlg.resize(min(1200, pix.width() + 40), min(800, pix.height() + 40))
        dlg.exec()
