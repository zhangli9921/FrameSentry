"""ReviewPanel all-hits wall + event mode (offscreen Qt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QListView

from framesentry.core.types import DetectionEvent, DetectionHit
from framesentry.ui.review_panel import MODE_ALL_HITS, MODE_EVENTS, ReviewPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_panel(tmp_path: Path, multi: bool = False):
    review = tmp_path / "rev"
    frames = review / "frames"
    frames.mkdir(parents=True)
    import cv2

    fname = "000001__00-00-01.000__FEMALE_BREAST_EXPOSED__0.90.jpg"
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)
    cv2.imwrite(str(frames / fname), img)

    hits = [
        DetectionHit(
            frame_index=1,
            timestamp_sec=1.0,
            timestamp_str="00-00-01.000",
            class_name="FEMALE_BREAST_EXPOSED",
            score=0.9,
            box=[1, 2, 3, 4],
            frame_filename=fname,
        )
    ]
    if multi:
        hits.append(
            DetectionHit(
                frame_index=1,
                timestamp_sec=1.0,
                timestamp_str="00-00-01.000",
                class_name="BUTTOCKS_EXPOSED",
                score=0.8,
                box=[5, 6, 7, 8],
                frame_filename=fname,
            )
        )
    events = [
        DetectionEvent(
            start_sec=1.0,
            end_sec=1.0,
            start_str="00-00-01.000",
            end_str="00-00-01.000",
            count=len(hits),
            classes=[h.class_name for h in hits],
            max_score=0.9,
            best_frame_index=1,
            best_frame_filename=fname,
            best_timestamp_sec=1.0,
            hit_indices=list(range(len(hits))),
        )
    ]
    panel = ReviewPanel()
    panel.load_from_memory(
        video_path="/v.mp4",
        review_dir=review,
        hits=hits,
        events=events,
    )
    return panel, fname


def test_review_panel_icon_mode_and_thumb(qapp, tmp_path: Path):
    panel, _ = _make_panel(tmp_path)
    assert panel.event_list.viewMode() == QListView.ViewMode.IconMode
    assert panel.frame_list.count() == 1
    item = panel.frame_list.item(0)
    assert not item.icon().isNull()
    assert panel.current_mode() == MODE_ALL_HITS


def test_multi_det_same_frame_one_tile(qapp, tmp_path: Path):
    panel, _ = _make_panel(tmp_path, multi=True)
    assert panel.frame_list.count() == 1
    text = panel.frame_list.item(0).text()
    assert "FEMALE_BREAST_EXPOSED" in text
    assert "BUTTOCKS_EXPOSED" in text


def test_mode_toggle_events(qapp, tmp_path: Path):
    panel, _ = _make_panel(tmp_path)
    panel.set_mode(MODE_EVENTS)
    assert panel.current_mode() == MODE_EVENTS
    assert panel.frame_list.count() == 1


def test_empty_hits_shows_message(qapp, tmp_path: Path):
    review = tmp_path / "empty"
    (review / "frames").mkdir(parents=True)
    panel = ReviewPanel()
    panel.load_from_memory(
        video_path="/v.mp4",
        review_dir=review,
        hits=[],
        events=[],
    )
    assert not panel.empty_label.isHidden()
    assert panel.frame_list.isHidden()
    assert "未检测到目标违规帧" in panel.empty_label.text()


def test_hit_list_hidden(qapp, tmp_path: Path):
    panel, _ = _make_panel(tmp_path)
    assert panel.hit_list.isHidden()
