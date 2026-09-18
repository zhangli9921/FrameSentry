"""MainWindow review open buttons — mocked openers, no real Explorer/browser."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from framesentry.core.types import DetectionEvent, DetectionHit
from framesentry.ui.main_window import MainWindow
from framesentry.ui.review_panel import MODE_ALL_HITS


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _prepare_review(tmp_path: Path) -> Path:
    review = tmp_path / "rev"
    frames = review / "frames"
    frames.mkdir(parents=True)
    import cv2

    fname = "000001__00-00-01.000__X__0.90.jpg"
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(frames / fname), img)
    (review / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    (review / "results.json").write_text(
        '{"video_path":"/v.mp4","hit_count":1,"event_count":1,"hits":[],"events":[],"meta":{}}',
        encoding="utf-8",
    )
    return review, fname


def test_open_buttons_call_platform_helpers(qapp, tmp_path: Path):
    review, fname = _prepare_review(tmp_path)
    win = MainWindow()
    hits = [
        DetectionHit(
            frame_index=1,
            timestamp_sec=1.0,
            timestamp_str="00-00-01.000",
            class_name="FEMALE_BREAST_EXPOSED",
            score=0.9,
            box=[0, 0, 1, 1],
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
    win.review.set_mode(MODE_ALL_HITS)
    win.review.load_from_memory(
        video_path="/v.mp4",
        review_dir=review,
        hits=hits,
        events=events,
    )
    assert win.review.current_frame_path() is not None

    with patch("framesentry.core.platform_open.open_html_in_browser") as open_html:
        win._on_open_review_page()
        open_html.assert_called_once()
        assert Path(open_html.call_args[0][0]).name == "index.html"

    with patch("framesentry.core.platform_open.open_directory") as open_dir:
        win._on_open_result_folder()
        open_dir.assert_called_once()

    with patch("framesentry.core.platform_open.reveal_in_file_manager") as reveal:
        win._on_locate_current_image()
        reveal.assert_called_once()

    with patch("framesentry.core.platform_open.open_path_with_default_app") as open_img:
        win._on_open_current_image()
        open_img.assert_called_once()

    with patch("framesentry.core.platform_open.open_logs_directory", return_value=tmp_path) as ol:
        win._on_open_logs()
        ol.assert_called_once()

    win.close()


def test_main_window_has_review_buttons(qapp):
    win = MainWindow()
    assert hasattr(win, "btn_open_review")
    assert hasattr(win, "btn_open_folder")
    assert hasattr(win, "btn_locate_image")
    assert hasattr(win, "btn_open_image")
    assert hasattr(win, "btn_open_logs")
    assert hasattr(win, "start_ort_probe")
    # __init__ must not finish with sync ORT "是/否" — probing label
    assert "检测中" in win.ort_label.text() or "ORT" in win.ort_label.text()
    win.close()


def test_open_review_missing_shows_error_no_crash(qapp):
    win = MainWindow()
    with patch.object(win, "_gui_error") as err:
        win._on_open_review_page()
        err.assert_called()
    win.close()
