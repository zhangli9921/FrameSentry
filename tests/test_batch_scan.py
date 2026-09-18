"""Batched detection: batch runs, order, cancel between batches, no fake-batch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from framesentry.core.config import TARGET_CLASSES
from framesentry.scanner.worker import ScanCancelled, ScanSettings, scan_video

_CLS = sorted(TARGET_CLASSES)[0]


class FakeCap:
    def __init__(self, frames, fps=25.0, count=None):
        self._frames = list(frames)
        self._i = 0
        self._fps = fps
        self._count = len(frames) if count is None else count
        self._opened = True

    def isOpened(self):
        return self._opened

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._count
        return 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        self._opened = False


def _frame():
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:] = (5, 10, 15)
    return arr


def _detector(*, supports_batch=True, batch_dim="batch"):
    detector = MagicMock()
    detector.detect_batch.side_effect = lambda imgs, batch_size=16: [[] for _ in imgs]
    detector.session = MagicMock()
    detector.session.get_providers.return_value = ["CUDAExecutionProvider"]
    detector.onnx_input_batch_dim.return_value = batch_dim
    detector.supports_batch_gt1.return_value = supports_batch
    return detector


def test_33_frames_batch_16_makes_3_runs(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    frames = [_frame() for _ in range(33)]
    detector = _detector()
    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=False,
        sample_fps=25.0,
        batch_size=16,
        device="gpu",
    )
    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=FakeCap(frames)):
        result = scan_video(str(video), detector, settings)
    assert result["meta"]["batch_runs"] == 3
    assert result["sampled_frames"] == 33
    assert detector.detect_batch.call_count == 3
    sizes = [len(c.args[0]) for c in detector.detect_batch.call_args_list]
    assert sizes == [16, 16, 1]


def test_batch_order_preserved(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    frames = [_frame() for _ in range(5)]
    detector = _detector()

    def detect_batch(imgs, batch_size=4):
        return [[{"class": _CLS, "score": 0.9, "box": [0, 0, 1, 1]}] for _ in imgs]

    detector.detect_batch.side_effect = detect_batch
    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=False,
        sample_fps=25.0,
        batch_size=4,
        device="gpu",
    )
    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=FakeCap(frames)):
        result = scan_video(str(video), detector, settings)
    idxs = [h.frame_index for h in result["hits"]]
    assert idxs == [0, 1, 2, 3, 4]


def test_cancel_between_batches(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    frames = [_frame() for _ in range(40)]
    detector = _detector()
    calls = {"n": 0}

    def detect_batch(imgs, batch_size=16):
        calls["n"] += 1
        return [[] for _ in imgs]

    detector.detect_batch.side_effect = detect_batch
    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=False,
        sample_fps=25.0,
        batch_size=16,
        device="gpu",
    )

    def should_cancel():
        return calls["n"] >= 1

    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=FakeCap(frames)):
        with pytest.raises(ScanCancelled):
            scan_video(str(video), detector, settings, should_cancel=should_cancel)
    assert calls["n"] >= 1


def test_fixed_batch1_does_not_fake_batch(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    frames = [_frame() for _ in range(5)]
    detector = _detector(supports_batch=False, batch_dim=1)
    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=False,
        sample_fps=25.0,
        batch_size=16,
        device="gpu",
    )
    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=FakeCap(frames)):
        result = scan_video(str(video), detector, settings)
    assert result["meta"]["batch_size"] == 1
    assert result["meta"]["onnx_supports_batch_gt1"] is False
    assert result["meta"]["batch_runs"] == 5


def test_scan_input_path_used_for_decode_review_uses_original(tmp_path: Path):
    original = tmp_path / "orig.flv"
    original.write_bytes(b"orig")
    intermediate = tmp_path / "orig.deadbeef.mp4"
    intermediate.write_bytes(b"mp4")
    frames = [_frame() for _ in range(3)]
    detector = _detector()
    opened: list[str] = []

    def capture_factory(path):
        opened.append(str(path))
        return FakeCap(frames)

    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=False,
        sample_fps=25.0,
        batch_size=2,
        device="gpu",
    )
    with patch("framesentry.scanner.worker.cv2.VideoCapture", side_effect=capture_factory):
        result = scan_video(
            str(original),
            detector,
            settings,
            scan_input_path=str(intermediate),
            preprocess_meta={
                "source_video_path": str(original),
                "scan_input_path": str(intermediate),
                "intermediate_path": str(intermediate),
                "preprocess_duration_sec": 0.5,
                "ffmpeg_exit_code": 0,
            },
        )
    assert opened == [str(intermediate)]
    assert result["meta"]["source_video_path"] == str(original)
    assert result["meta"]["scan_input_path"] == str(intermediate)
    assert Path(result["review_dir"]).is_dir()
    assert "orig" in Path(result["review_dir"]).name


def test_indeterminate_progress_when_frame_count_unknown(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    frames = [_frame() for _ in range(6)]
    detector = _detector()
    progress_vals: list[float] = []

    def on_progress(pct, msg):
        progress_vals.append(pct)

    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=False,
        sample_fps=25.0,
        batch_size=4,
        device="gpu",
    )
    with patch(
        "framesentry.scanner.worker.cv2.VideoCapture",
        return_value=FakeCap(frames, count=0),
    ):
        scan_video(str(video), detector, settings, on_progress=on_progress)
    # Should report -1 (indeterminate) at least once, never stuck bogus 0% only
    assert any(p < 0 for p in progress_vals[:-1] or progress_vals)
