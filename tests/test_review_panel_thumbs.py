"""ReviewPanel thumbnail browser smoke (offscreen Qt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QListView

from framesentry.core.types import DetectionEvent, DetectionHit
from framesentry.ui.review_panel import ReviewPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_review_panel_icon_mode_and_thumb(qapp, tmp_path: Path):
    review = tmp_path / "rev"
    frames = review / "frames"
    frames.mkdir(parents=True)
    import cv2

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)
    fname = "000001__00-00-01.000__FEMALE_BREAST_EXPOSED__0.90.jpg"
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
    events = [
        DetectionEvent(
            start_sec=1.0,
            end_sec=1.0,
            start_str="00-00-01.000",
            end_str="00-00-01.000",
            count=1,
            classes=["FEMALE_BREAST_EXPOSED"],
            max_score=0.9,
            best_frame_index=1,
            best_frame_filename=fname,
            best_timestamp_sec=1.0,
            hit_indices=[0],
        )
    ]

    panel = ReviewPanel()
    panel.load_from_memory(
        video_path="/v.mp4",
        review_dir=review,
        hits=hits,
        events=events,
    )
    assert panel.event_list.viewMode() == QListView.ViewMode.IconMode
    assert panel.event_list.count() == 1
    item = panel.event_list.item(0)
    assert not item.icon().isNull()
