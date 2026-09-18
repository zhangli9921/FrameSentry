"""Timing boundaries: preprocess_tensor / inference / postprocess / frame_save."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from framesentry.core.config import TARGET_CLASSES
from framesentry.scanner.worker import ScanSettings, scan_video

_CLS = sorted(TARGET_CLASSES)[0]


class FakeCap:
    def __init__(self, frames, fps=25.0):
        self._frames = list(frames)
        self._i = 0
        self._fps = fps
        self._opened = True

    def isOpened(self):
        return self._opened

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        return 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        self._opened = False


def test_meta_includes_split_timing_fields(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]

    detector = MagicMock()
    detector.onnx_input_batch_dim.return_value = "batch"
    detector.supports_batch_gt1.return_value = True
    detector.session = MagicMock()
    detector.session.get_providers.return_value = ["CUDAExecutionProvider"]

    def detect_batch(imgs, batch_size=4):
        detector.last_batch_timing = {
            "preprocess_tensor_sec": 0.01,
            "inference_sec": 0.02,
            "postprocess_sec": 0.003,
        }
        return [[{"class": _CLS, "score": 0.9, "box": [0, 0, 1, 1]}] for _ in imgs]

    detector.detect_batch.side_effect = detect_batch
    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=True,
        sample_fps=25.0,
        batch_size=4,
        device="gpu",
    )
    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=FakeCap(frames)):
        with patch("framesentry.scanner.worker.annotate_and_save_frame"):
            result = scan_video(str(video), detector, settings)
    meta = result["meta"]
    assert "preprocess_tensor_sec" in meta
    assert "inference_sec" in meta
    assert "postprocess_sec" in meta
    assert "frame_save_sec" in meta
    assert "total_sec" in meta
    assert "batch_runs" in meta
    assert "sampled_frames" in meta
    assert "ms_per_batch" in meta
    assert "inference_ms_per_frame" in meta
    assert "active_ort_providers" in meta
    assert meta["preprocess_tensor_sec"] >= 0.01
    assert meta["inference_sec"] >= 0.02
    # inference_ms_per_frame is pure session.run based
    assert meta["inference_ms_per_frame"] == round(
        meta["inference_sec"] * 1000.0 / meta["sampled_frames"], 3
    )
