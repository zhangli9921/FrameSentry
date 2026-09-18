"""sampled_frames==0 must raise UnreadableVideoError (FAILED), not COMPLETED."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from framesentry.scanner.worker import ScanSettings, UnreadableVideoError, scan_video


class _FakeCap:
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 100
        return 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame

    def release(self):
        self._opened = False


def test_zero_usable_frames_raises_unreadable(tmp_path: Path):
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"fake")
    detector = MagicMock()
    settings = ScanSettings(output_root=str(tmp_path / "out"), save_frames=False)

    fake = _FakeCap([None, np.zeros((0, 0, 3), dtype=np.uint8)])

    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=fake):
        with pytest.raises(UnreadableVideoError) as ei:
            scan_video(str(video), detector, settings)
    msg = str(ei.value).lower()
    assert "0 usable" in msg or "unreadable" in msg
    detector.detect.assert_not_called()


def test_zero_frames_eof_raises(tmp_path: Path):
    video = tmp_path / "nof.mp4"
    video.write_bytes(b"fake")
    detector = MagicMock()
    settings = ScanSettings(output_root=str(tmp_path / "out"), save_frames=False)
    fake = _FakeCap([])

    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=fake):
        with pytest.raises(UnreadableVideoError):
            scan_video(str(video), detector, settings)


def test_detect_errors_counted_separately(tmp_path: Path):
    video = tmp_path / "ok.mp4"
    video.write_bytes(b"fake")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    fake = _FakeCap([frame, frame])

    detector = MagicMock()
    detector.detect.side_effect = RuntimeError("boom")

    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=False,
        sample_fps=25.0,
    )

    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=fake):
        with pytest.raises(UnreadableVideoError) as ei:
            scan_video(str(video), detector, settings)
    assert "detect_errors" in str(ei.value)
