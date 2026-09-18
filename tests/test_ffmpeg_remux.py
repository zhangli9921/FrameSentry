"""FFmpeg resolve + stream-copy remux (mocked subprocess — no real FFmpeg required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from framesentry.core.ffmpeg import (
    FFmpegProcessHandle,
    RemuxCancelled,
    RemuxError,
    build_remux_argv,
    remux_stream_copy,
    resolve_ffmpeg,
)


def test_resolve_ffmpeg_env(tmp_path: Path):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    assert resolve_ffmpeg(env={"FRAMESENTRY_FFMPEG": str(fake)}) == str(fake.resolve())


def test_resolve_ffmpeg_which(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "framesentry.core.ffmpeg.shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    assert resolve_ffmpeg(env={}) == "/usr/bin/ffmpeg"


def test_resolve_ffmpeg_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("framesentry.core.ffmpeg.shutil.which", lambda _: None)
    assert resolve_ffmpeg(env={}, nearby_roots=[]) is None


def test_build_remux_argv_is_stream_copy_no_shell():
    argv = build_remux_argv("/bin/ffmpeg", "/src/a.flv", "/dst/a.part.mp4")
    assert argv[0] == "/bin/ffmpeg"
    assert "-c:v" in argv and argv[argv.index("-c:v") + 1] == "copy"
    assert "-an" in argv and "-sn" in argv and "-dn" in argv
    assert "-map" in argv
    joined = " ".join(argv)
    assert "libx264" not in joined
    assert "h264_nvenc" not in joined


def _fake_proc(returncode: int = 0, stderr_bytes: bytes = b"ok"):
    class FakeProc:
        def __init__(self):
            self.returncode = returncode
            self.stderr = MagicMock()
            self.stderr.read.side_effect = [stderr_bytes, b""]

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    return FakeProc()


def test_remux_atomic_rename_and_cleanup_part(tmp_path: Path):
    src = tmp_path / "src.flv"
    src.write_bytes(b"fake-flv")
    dst = tmp_path / "out.mp4"
    part = dst.with_name(dst.stem + ".part.mp4")

    def fake_popen(argv, **kwargs):
        assert kwargs.get("shell") in (False, None)
        part.write_bytes(b"mp4data")
        return _fake_proc(0)

    with patch("framesentry.core.ffmpeg.subprocess.Popen", side_effect=fake_popen):
        result = remux_stream_copy(src, dst, ffmpeg_bin="/bin/ffmpeg")
    assert dst.is_file()
    assert not part.exists()
    assert result.exit_code == 0
    assert src.is_file()


def test_remux_failure_deletes_part_keeps_source(tmp_path: Path):
    src = tmp_path / "src.flv"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"
    part = dst.with_name(dst.stem + ".part.mp4")

    def fake_popen(argv, **kwargs):
        part.write_bytes(b"partial")
        return _fake_proc(1, b"codec not supported")

    with patch("framesentry.core.ffmpeg.subprocess.Popen", side_effect=fake_popen):
        with pytest.raises(RemuxError) as ei:
            remux_stream_copy(src, dst, ffmpeg_bin="/bin/ffmpeg")
    assert not part.exists()
    assert not dst.exists()
    assert src.is_file()
    assert ei.value.exit_code == 1


def test_remux_cancel_cleans_part(tmp_path: Path):
    src = tmp_path / "src.flv"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"
    part = dst.with_name(dst.stem + ".part.mp4")
    handle = FFmpegProcessHandle()

    class SlowProc:
        def __init__(self):
            self.returncode = None
            self.stderr = MagicMock()
            self.stderr.read.side_effect = lambda n=4096: b""

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.returncode = -9
            return -9

        def terminate(self):
            self.returncode = -9

        def kill(self):
            self.returncode = -9

    def fake_popen(argv, **kwargs):
        part.write_bytes(b"partial")
        return SlowProc()

    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] >= 2

    with patch("framesentry.core.ffmpeg.subprocess.Popen", side_effect=fake_popen):
        with pytest.raises(RemuxCancelled):
            remux_stream_copy(
                src,
                dst,
                ffmpeg_bin="/bin/ffmpeg",
                should_cancel=should_cancel,
                process_handle=handle,
            )
    assert not part.exists()
    assert src.is_file()
