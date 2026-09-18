"""Pipeline concurrency: ≤1 FFmpeg; only next-file preprocess while scanning."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from framesentry.core.types import VideoStatus
from framesentry.detectors.nudenet_backend import GpuProviderUnavailableError
from framesentry.scanner.pipeline import RemuxPipeline
from framesentry.scanner.preprocess import PreprocessFailed, PreprocessResult
from framesentry.scanner.worker import ScanSettings
from framesentry.ui.scan_controller import ScanWorker


def test_only_one_ffmpeg_and_only_next_preprocess(tmp_path: Path):
    paths = [str(tmp_path / f"v{i}.flv") for i in range(4)]
    for p in paths:
        Path(p).write_bytes(b"x")

    active = {"n": 0, "max": 0}
    lock = threading.Lock()
    # Overlap tracking: which path is scanning / preprocessing right now
    current_scan: dict[str, str | None] = {"path": None}
    bad_overlaps: list[tuple[str, str]] = []

    def fake_preprocess(source_path, intermediate_dir, **kwargs):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
            scan = current_scan["path"]
            if scan is not None:
                si = paths.index(scan)
                pi = paths.index(str(source_path))
                if pi != si + 1:
                    bad_overlaps.append((scan, str(source_path)))
        try:
            time.sleep(0.05)
            dst = Path(intermediate_dir) / (Path(source_path).stem + ".mp4")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"mp4")
            return PreprocessResult(
                source_path=str(source_path),
                intermediate_path=str(dst),
                duration_sec=0.05,
                exit_code=0,
                ffmpeg_path="/bin/ffmpeg",
                status="READY",
            )
        finally:
            with lock:
                active["n"] -= 1

    def fake_scan(video_path, detector, settings, **kwargs):
        with lock:
            current_scan["path"] = str(video_path)
        try:
            time.sleep(0.08)
            return {
                "review_dir": str(tmp_path / "rev"),
                "hits": [],
                "events": [],
                "hit_count": 0,
                "meta": {},
            }
        finally:
            with lock:
                if current_scan["path"] == str(video_path):
                    current_scan["path"] = None

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
        ).run(paths)
    finally:
        mp.undo()

    assert active["max"] <= 1, f"concurrent ffmpeg peaked at {active['max']}"
    assert bad_overlaps == [], f"non-next preprocess during scan: {bad_overlaps}"


def test_preprocess_fail_continues_queue(tmp_path: Path):
    paths = [str(tmp_path / f"v{i}.flv") for i in range(3)]
    for p in paths:
        Path(p).write_bytes(b"x")

    def fake_preprocess(source_path, intermediate_dir, **kwargs):
        if source_path.endswith("v1.flv"):
            raise PreprocessFailed("boom", exit_code=2, stderr_tail="x")
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

    finished: list[tuple[str, str]] = []

    def on_finished(path, status, result):
        finished.append((path, status))

    def fake_scan(video_path, detector, settings, **kwargs):
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
            on_finished=on_finished,
        ).run(paths)
    finally:
        mp.undo()

    statuses = {p: s for p, s in finished}
    assert statuses[paths[1]] == VideoStatus.FAILED.value
    assert statuses[paths[0]] == VideoStatus.COMPLETED.value
    assert statuses[paths[2]] == VideoStatus.COMPLETED.value


def test_no_silent_cpu_fallback_on_gpu_detector_init(tmp_path: Path):
    worker = ScanWorker()
    settings = ScanSettings(device="gpu", output_root=str(tmp_path / "out"))
    paths = [str(tmp_path / "a.flv")]
    Path(paths[0]).write_bytes(b"x")
    finished: list[tuple[str, str]] = []
    worker.job_finished.connect(lambda p, s, r: finished.append((p, s)))
    worker.configure(
        paths,
        settings,
        lambda: (_ for _ in ()).throw(GpuProviderUnavailableError("no cuda")),
    )
    worker.run()
    assert finished
    assert finished[0][1] == VideoStatus.FAILED.value
