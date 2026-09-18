"""Cancel-current must not kill ahead FFmpeg or leak cancel flag into next files."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from framesentry.core.ffmpeg import RemuxCancelled
from framesentry.core.types import VideoStatus
from framesentry.scanner.pipeline import RemuxPipeline
from framesentry.scanner.preprocess import PreprocessResult
from framesentry.scanner.worker import ScanCancelled, ScanSettings
from framesentry.ui.scan_controller import ScanWorker


def _prep_result(source: str, intermediate_dir: str) -> PreprocessResult:
    dst = Path(intermediate_dir) / (Path(source).stem + ".mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"mp4")
    return PreprocessResult(
        source_path=str(source),
        intermediate_path=str(dst),
        duration_sec=0.01,
        exit_code=0,
        ffmpeg_path="/bin/ffmpeg",
        status="READY",
    )


def test_cancel_current_during_preprocess_a_does_not_cancel_b(tmp_path: Path):
    """cancel_current while PREPROCESS A: kill only A's ffmpeg; B continues."""
    paths = [str(tmp_path / f"v{i}.flv") for i in range(3)]
    for p in paths:
        Path(p).write_bytes(b"x")

    flags = {"cancel_current": False, "cancel_queue": False}
    killed_roles: list[str] = []
    prep_roles: dict[str, str] = {}
    finished: list[tuple[str, str]] = []
    lock = threading.Lock()
    a_started = threading.Event()

    def fake_preprocess(source_path, intermediate_dir, **kwargs):
        role = "current" if str(source_path) == paths[0] else "ahead"
        with lock:
            prep_roles[str(source_path)] = role
        should_cancel = kwargs.get("should_cancel")
        if str(source_path) == paths[0]:
            a_started.set()
            # Wait until cancel-current is requested
            for _ in range(200):
                if should_cancel and should_cancel():
                    raise RemuxCancelled("cancelled A")
                time.sleep(0.01)
            raise RemuxCancelled("timeout waiting cancel")
        # B / C: must NOT see cancel-current
        time.sleep(0.02)
        if should_cancel and should_cancel():
            raise RemuxCancelled(f"unexpected cancel during {source_path}")
        return _prep_result(str(source_path), intermediate_dir)

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

    clear_calls = {"n": 0}

    def clear_cancel_current():
        flags["cancel_current"] = False
        clear_calls["n"] += 1

    pipe = RemuxPipeline(
        intermediate_dir=str(tmp_path / "inter"),
        settings=ScanSettings(output_root=str(tmp_path / "out"), save_frames=False),
        detector=MagicMock(),
        ffmpeg_bin="/bin/ffmpeg",
        should_cancel_current=lambda: flags["cancel_current"],
        should_cancel_queue=lambda: flags["cancel_queue"],
        clear_cancel_current=clear_cancel_current,
        on_finished=lambda p, s, r: finished.append((p, s)),
    )

    # Wrap cancel_current_ffmpeg to record role
    orig = pipe.cancel_current_ffmpeg

    def wrapped_cancel():
        with lock:
            killed_roles.append(pipe._ffmpeg_role)
        return orig()

    pipe.cancel_current_ffmpeg = wrapped_cancel  # type: ignore[method-assign]

    def trigger():
        assert a_started.wait(2.0)
        time.sleep(0.05)
        flags["cancel_current"] = True
        pipe.cancel_current_ffmpeg()

    t = threading.Thread(target=trigger, daemon=True)
    try:
        t.start()
        pipe.run(paths)
        t.join(timeout=2.0)
    finally:
        mp.undo()

    statuses = {p: s for p, s in finished}
    assert statuses[paths[0]] == VideoStatus.CANCELLED.value
    assert statuses[paths[1]] == VideoStatus.COMPLETED.value
    assert statuses[paths[2]] == VideoStatus.COMPLETED.value
    assert clear_calls["n"] >= 1, "cancel-current flag must be cleared before next file"
    assert flags["cancel_current"] is False


def test_cancel_current_during_scan_a_does_not_kill_ahead_b(tmp_path: Path):
    """cancel_current while SCAN A: cancel A only; ahead B ffmpeg must survive."""
    paths = [str(tmp_path / f"v{i}.flv") for i in range(3)]
    for p in paths:
        Path(p).write_bytes(b"x")

    flags = {"cancel_current": False, "cancel_queue": False}
    finished: list[tuple[str, str]] = []
    ahead_killed = {"v": False}
    a_scanning = threading.Event()
    b_prep_started = threading.Event()
    b_prep_finished = threading.Event()
    cancel_seen_during_b = {"v": False}
    clear_calls = {"n": 0}

    def clear_cancel_current():
        flags["cancel_current"] = False
        clear_calls["n"] += 1

    def fake_preprocess(source_path, intermediate_dir, **kwargs):
        should_cancel = kwargs.get("should_cancel")
        if str(source_path) == paths[1]:
            b_prep_started.set()
            # Simulate long ahead preprocess overlapping A's scan cancel
            for _ in range(100):
                if should_cancel and should_cancel():
                    cancel_seen_during_b["v"] = True
                    raise RemuxCancelled("B should not be cancelled by cancel-current")
                if a_scanning.is_set() and flags["cancel_current"]:
                    # Give cancel-current a window while B is still remuxing
                    time.sleep(0.05)
                time.sleep(0.01)
            b_prep_finished.set()
            return _prep_result(str(source_path), intermediate_dir)
        return _prep_result(str(source_path), intermediate_dir)

    def fake_scan(video_path, detector, settings, **kwargs):
        should_cancel = kwargs.get("should_cancel")
        if str(video_path) == paths[0]:
            a_scanning.set()
            # Wait for B ahead preprocess to start, then request cancel-current
            assert b_prep_started.wait(2.0)
            flags["cancel_current"] = True
            # Pipeline's cancel_current_ffmpeg should refuse to kill ahead
            pipe.cancel_current_ffmpeg()
            for _ in range(50):
                if should_cancel and should_cancel():
                    raise ScanCancelled("cancel A")
                time.sleep(0.01)
            raise ScanCancelled("cancel A timeout")
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

    pipe = RemuxPipeline(
        intermediate_dir=str(tmp_path / "inter"),
        settings=ScanSettings(output_root=str(tmp_path / "out"), save_frames=False),
        detector=MagicMock(),
        ffmpeg_bin="/bin/ffmpeg",
        should_cancel_current=lambda: flags["cancel_current"],
        should_cancel_queue=lambda: flags["cancel_queue"],
        clear_cancel_current=clear_cancel_current,
        on_finished=lambda p, s, r: finished.append((p, s)),
    )

    orig_kill = pipe.kill_ffmpeg

    def tracking_kill():
        # If kill happens while role is ahead during cancel-current, that's a bug
        if pipe._ffmpeg_role == "ahead" and not flags["cancel_queue"]:
            ahead_killed["v"] = True
        return orig_kill()

    pipe.kill_ffmpeg = tracking_kill  # type: ignore[method-assign]

    try:
        pipe.run(paths)
    finally:
        mp.undo()

    statuses = {p: s for p, s in finished}
    assert statuses[paths[0]] == VideoStatus.CANCELLED.value
    assert statuses[paths[1]] == VideoStatus.COMPLETED.value
    assert statuses[paths[2]] == VideoStatus.COMPLETED.value
    assert ahead_killed["v"] is False
    assert cancel_seen_during_b["v"] is False
    assert clear_calls["n"] >= 1
    assert flags["cancel_current"] is False
    assert b_prep_finished.is_set() or statuses[paths[1]] == VideoStatus.COMPLETED.value


def test_cancel_queue_kills_ffmpeg_and_cancels_remaining(tmp_path: Path):
    paths = [str(tmp_path / f"v{i}.flv") for i in range(3)]
    for p in paths:
        Path(p).write_bytes(b"x")

    flags = {"cancel_current": False, "cancel_queue": False}
    finished: list[tuple[str, str]] = []
    kill_count = {"n": 0}

    def fake_preprocess(source_path, intermediate_dir, **kwargs):
        should_cancel = kwargs.get("should_cancel")
        if str(source_path) == paths[0]:
            return _prep_result(str(source_path), intermediate_dir)
        # Block so queue cancel can fire during B preprocess / A scan
        for _ in range(200):
            if should_cancel and should_cancel():
                raise RemuxCancelled("queue cancel")
            time.sleep(0.01)
        return _prep_result(str(source_path), intermediate_dir)

    def fake_scan(video_path, detector, settings, **kwargs):
        should_cancel = kwargs.get("should_cancel")
        flags["cancel_queue"] = True
        flags["cancel_current"] = True
        pipe.kill_ffmpeg()
        for _ in range(50):
            if should_cancel and should_cancel():
                raise ScanCancelled("queue")
            time.sleep(0.01)
        raise ScanCancelled("queue timeout")

    import framesentry.scanner.pipeline as pipe_mod

    mp = pytest.MonkeyPatch()
    mp.setattr(pipe_mod, "preprocess_video", fake_preprocess)
    mp.setattr(pipe_mod, "scan_video", fake_scan)

    pipe = RemuxPipeline(
        intermediate_dir=str(tmp_path / "inter"),
        settings=ScanSettings(output_root=str(tmp_path / "out"), save_frames=False),
        detector=MagicMock(),
        ffmpeg_bin="/bin/ffmpeg",
        should_cancel_current=lambda: flags["cancel_current"],
        should_cancel_queue=lambda: flags["cancel_queue"],
        on_finished=lambda p, s, r: finished.append((p, s)),
    )
    orig = pipe.kill_ffmpeg

    def counting_kill():
        kill_count["n"] += 1
        return orig()

    pipe.kill_ffmpeg = counting_kill  # type: ignore[method-assign]

    try:
        pipe.run(paths)
    finally:
        mp.undo()

    statuses = {p: s for p, s in finished}
    assert statuses[paths[0]] == VideoStatus.CANCELLED.value
    assert statuses[paths[1]] == VideoStatus.CANCELLED.value
    assert statuses[paths[2]] == VideoStatus.CANCELLED.value
    assert kill_count["n"] >= 1


def test_scan_worker_clear_cancel_current_resets_flag():
    worker = ScanWorker()
    worker._cancel_current = True
    worker.clear_cancel_current()
    assert worker._flags()[0] is False
