"""Never unconditionally reuse stale intermediate MP4; remux failure must not scan old final."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from framesentry.core.ffmpeg import RemuxError, RemuxResult
from framesentry.scanner.preprocess import (
    PreprocessFailed,
    intermediate_mp4_path,
    preprocess_video,
)


def test_default_always_remuxes_even_if_final_exists(tmp_path: Path):
    src = tmp_path / "clip.flv"
    src.write_bytes(b"source-v1")
    inter = tmp_path / "inter"
    dst = intermediate_mp4_path(src, inter)
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"OLD-FINAL")

    calls = {"n": 0}

    def fake_remux(s, d, **kwargs):
        calls["n"] += 1
        # remux_stream_copy writes part then replaces; simulate final content change
        Path(d).parent.mkdir(parents=True, exist_ok=True)
        Path(d).write_bytes(b"NEW-FINAL")
        return RemuxResult(
            source_path=str(s),
            intermediate_path=str(d),
            duration_sec=0.1,
            exit_code=0,
            ffmpeg_path="/bin/ffmpeg",
            stderr_tail="",
        )

    with patch("framesentry.scanner.preprocess.remux_stream_copy", side_effect=fake_remux):
        result = preprocess_video(src, inter, ffmpeg_bin="/bin/ffmpeg")
    assert calls["n"] == 1, "default reuse_existing=False must remux even when final exists"
    assert Path(result.intermediate_path).read_bytes() == b"NEW-FINAL"
    assert src.is_file() and src.read_bytes() == b"source-v1"


def test_source_mtime_or_content_change_triggers_ffmpeg_again(tmp_path: Path):
    src = tmp_path / "clip.flv"
    src.write_bytes(b"v1")
    inter = tmp_path / "inter"
    calls = {"n": 0}

    def fake_remux(s, d, **kwargs):
        calls["n"] += 1
        Path(d).parent.mkdir(parents=True, exist_ok=True)
        Path(d).write_bytes(f"out-{calls['n']}".encode())
        return RemuxResult(
            source_path=str(s),
            intermediate_path=str(d),
            duration_sec=0.05,
            exit_code=0,
            ffmpeg_path="/bin/ffmpeg",
            stderr_tail="",
        )

    with patch("framesentry.scanner.preprocess.remux_stream_copy", side_effect=fake_remux):
        preprocess_video(src, inter, ffmpeg_bin="/bin/ffmpeg")
        src.write_bytes(b"v2-changed")
        preprocess_video(src, inter, ffmpeg_bin="/bin/ffmpeg")
    assert calls["n"] == 2


def test_remux_fail_with_old_final_present_raises_and_does_not_return_old(tmp_path: Path):
    src = tmp_path / "clip.flv"
    src.write_bytes(b"source")
    inter = tmp_path / "inter"
    dst = intermediate_mp4_path(src, inter)
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"STALE-FINAL")

    with patch(
        "framesentry.scanner.preprocess.remux_stream_copy",
        side_effect=RemuxError("boom", exit_code=69, stderr_tail="bad"),
    ):
        with pytest.raises(PreprocessFailed) as ei:
            preprocess_video(src, inter, ffmpeg_bin="/bin/ffmpeg")
        assert ei.value.exit_code == 69

    # Old final may remain on disk, but preprocess must not return it as READY.
    assert dst.is_file() and dst.read_bytes() == b"STALE-FINAL"
    assert src.is_file() and src.read_bytes() == b"source"


def test_reuse_existing_true_still_skips_when_explicitly_requested(tmp_path: Path):
    src = tmp_path / "clip.flv"
    src.write_bytes(b"source")
    inter = tmp_path / "inter"
    dst = intermediate_mp4_path(src, inter)
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"CACHED")

    with patch("framesentry.scanner.preprocess.remux_stream_copy") as remux:
        result = preprocess_video(
            src, inter, ffmpeg_bin="/bin/ffmpeg", reuse_existing=True
        )
        remux.assert_not_called()
    assert Path(result.intermediate_path).read_bytes() == b"CACHED"
