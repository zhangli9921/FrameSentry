"""Intermediate MP4 naming, paths, and preprocess edge cases (mocked FFmpeg)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from framesentry.core.ffmpeg import FFmpegNotFoundError, RemuxError, RemuxResult
from framesentry.scanner.preprocess import (
    PreprocessFailed,
    default_intermediate_dir,
    ensure_intermediate_dir,
    intermediate_mp4_name,
    intermediate_mp4_path,
    preprocess_video,
)
from framesentry.storage.paths import path_stable_id, review_dir_for_video


def test_default_intermediate_dir_is_home_framesentry_intermediate():
    d = default_intermediate_dir()
    assert d.name == "FrameSentryIntermediate"
    assert d.parent == Path.home()


def test_intermediate_name_same_stem_different_paths(tmp_path: Path):
    a = tmp_path / "dir a" / "clip.flv"
    b = tmp_path / "dir b" / "clip.flv"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    na = intermediate_mp4_name(a)
    nb = intermediate_mp4_name(b)
    assert na.startswith("clip.")
    assert nb.startswith("clip.")
    assert na != nb
    assert na.endswith(".mp4")
    assert path_stable_id(a) in na


def test_unicode_and_spaces_in_paths(tmp_path: Path):
    src = tmp_path / "视频 文件" / "测试 clip.flv"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    inter = tmp_path / "中间 dir"
    path = intermediate_mp4_path(src, inter)
    assert path.parent == inter
    assert path.name.endswith(".mp4")


def test_unc_style_path_stable_id_differs():
    p1 = r"\\server\share\a\clip.flv"
    p2 = r"\\server\share\b\clip.flv"
    assert path_stable_id(p1) != path_stable_id(p2)
    assert intermediate_mp4_name(p1) != intermediate_mp4_name(p2)


def test_review_id_based_on_original_not_intermediate(tmp_path: Path):
    src = tmp_path / "orig.flv"
    src.write_bytes(b"x")
    inter = tmp_path / "orig.abc123.mp4"
    inter.write_bytes(b"y")
    out = tmp_path / "out"
    r1 = review_dir_for_video(src, out)
    r2 = review_dir_for_video(inter, out)
    assert r1 != r2


def test_ensure_unwritable_dir_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    d = tmp_path / "readonly"
    d.mkdir()
    real_write = Path.write_text

    def selective_write(self, *a, **k):
        if ".framesentry_write_probe" in self.name:
            raise OSError("readonly")
        return real_write(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", selective_write)
    with pytest.raises(OSError):
        ensure_intermediate_dir(d)


def test_preprocess_missing_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / "a.flv"
    src.write_bytes(b"x")
    monkeypatch.setattr("framesentry.scanner.preprocess.resolve_ffmpeg", lambda: None)
    with pytest.raises(FFmpegNotFoundError):
        preprocess_video(src, tmp_path / "inter", ffmpeg_bin=None)


def test_preprocess_success_mocked(tmp_path: Path):
    src = tmp_path / "a.flv"
    src.write_bytes(b"x")
    inter_dir = tmp_path / "inter"

    def fake_remux(s, d, **kwargs):
        Path(d).parent.mkdir(parents=True, exist_ok=True)
        Path(d).write_bytes(b"mp4")
        return RemuxResult(
            source_path=str(s),
            intermediate_path=str(d),
            duration_sec=0.1,
            exit_code=0,
            ffmpeg_path="/bin/ffmpeg",
            stderr_tail="",
        )

    with patch("framesentry.scanner.preprocess.remux_stream_copy", side_effect=fake_remux):
        result = preprocess_video(src, inter_dir, ffmpeg_bin="/bin/ffmpeg")
    assert result.status == "READY"
    assert Path(result.intermediate_path).is_file()
    assert src.is_file()


def test_preprocess_fail_raises_preprocess_failed(tmp_path: Path):
    src = tmp_path / "a.flv"
    src.write_bytes(b"x")
    with patch(
        "framesentry.scanner.preprocess.remux_stream_copy",
        side_effect=RemuxError("fail", exit_code=69, stderr_tail="bad codec"),
    ):
        with pytest.raises(PreprocessFailed) as ei:
            preprocess_video(src, tmp_path / "inter", ffmpeg_bin="/bin/ffmpeg")
        assert ei.value.exit_code == 69
    assert src.is_file()
