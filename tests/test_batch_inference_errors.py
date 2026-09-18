"""Batch inference errors must FAIL the current video (no silent skip)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from framesentry.core.types import VideoStatus
from framesentry.scanner.pipeline import RemuxPipeline
from framesentry.scanner.preprocess import PreprocessResult
from framesentry.scanner.worker import ScanSettings, scan_video


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
    arr[:] = (1, 2, 3)
    return arr


def test_second_batch_exception_fails_video_third_not_success(tmp_path: Path):
    """33 frames batch=16: 2nd batch raises → video fails; 3rd batch not a success completion."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    frames = [_frame() for _ in range(33)]
    detector = MagicMock()
    detector.onnx_input_batch_dim.return_value = "batch"
    detector.supports_batch_gt1.return_value = True
    detector.session = MagicMock()
    detector.session.get_providers.return_value = ["CUDAExecutionProvider"]

    calls = {"n": 0}

    def detect_batch(imgs, batch_size=16):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("ORT boom on batch 2")
        return [[] for _ in imgs]

    detector.detect_batch.side_effect = detect_batch
    settings = ScanSettings(
        output_root=str(tmp_path / "out"),
        save_frames=False,
        sample_fps=25.0,
        batch_size=16,
        device="gpu",
    )
    with patch("framesentry.scanner.worker.cv2.VideoCapture", return_value=FakeCap(frames)):
        with pytest.raises(RuntimeError, match="ORT boom"):
            scan_video(str(video), detector, settings)
    assert calls["n"] == 2, "must stop at failing 2nd batch; 3rd must not run"
    # No successful completion artifact
    assert not any((tmp_path / "out").rglob("results.json"))


def test_batch_error_marks_video_failed_queue_continues(tmp_path: Path):
    paths = [str(tmp_path / f"v{i}.flv") for i in range(2)]
    for p in paths:
        Path(p).write_bytes(b"x")

    finished: list[tuple[str, str]] = []

    def fake_preprocess(source_path, intermediate_dir, **kwargs):
        dst = Path(intermediate_dir) / (Path(source_path).stem + ".mp4")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"mp4")
        return PreprocessResult(
            source_path=str(source_path),
            intermediate_path=str(dst),
            duration_sec=0.01,
            exit_code=0,
            ffmpeg_path="/bin/ffmpeg",
            status="READY",
        )

    def fake_scan(video_path, detector, settings, **kwargs):
        if str(video_path).endswith("v0.flv"):
            raise RuntimeError("batch inference failed")
        return {
            "review_dir": str(tmp_path / "r"),
            "hits": [],
            "events": [],
            "hit_count": 0,
            "meta": {},
        }

    import framesentry.scanner.pipeline as pipe_mod

    mp = pytest.MonkeyPatch()
    mp.setattr(pipe_mod, "preprocess_video", fake_preprocess)
    mp.setattr(pipe_mod, "scan_video", fake_scan)
    try:
        RemuxPipeline(
            intermediate_dir=str(tmp_path / "inter"),
            settings=ScanSettings(output_root=str(tmp_path / "out"), save_frames=False),
            detector=MagicMock(),
            ffmpeg_bin="/bin/ffmpeg",
            should_cancel_current=lambda: False,
            should_cancel_queue=lambda: False,
            on_finished=lambda p, s, r: finished.append((p, s)),
        ).run(paths)
    finally:
        mp.undo()

    statuses = {p: s for p, s in finished}
    assert statuses[paths[0]] == VideoStatus.FAILED.value
    assert statuses[paths[1]] == VideoStatus.COMPLETED.value
