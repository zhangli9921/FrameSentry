"""Video path discovery / DnD shared API."""

from pathlib import Path
from unittest.mock import patch

import pytest

from framesentry.core.input_discovery import (
    DiscoveryResult,
    discover_videos,
    is_supported_video,
    normalize_video_key,
)


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def test_single_file(tmp_path: Path):
    f = _touch(tmp_path / "a.mp4")
    r = discover_videos([f])
    assert r.added == 1
    assert len(r.videos) == 1
    assert Path(r.videos[0]).name == "a.mp4"


def test_multi_file(tmp_path: Path):
    files = [_touch(tmp_path / n) for n in ("a.mp4", "b.mkv", "c.mov")]
    r = discover_videos(files)
    assert r.added == 3


def test_recursive_folder(tmp_path: Path):
    _touch(tmp_path / "root.avi")
    _touch(tmp_path / "sub" / "nested.webm")
    _touch(tmp_path / "sub" / "ignore.txt")
    r = discover_videos([tmp_path])
    assert r.added == 2
    names = {Path(v).name for v in r.videos}
    assert names == {"root.avi", "nested.webm"}


def test_mixed_files_and_dirs(tmp_path: Path):
    f = _touch(tmp_path / "one.flv")
    d = tmp_path / "folder"
    _touch(d / "two.mp4")
    r = discover_videos([f, d])
    assert r.added == 2


def test_unsupported_extension_filter(tmp_path: Path):
    _touch(tmp_path / "pic.jpg")
    _touch(tmp_path / "doc.pdf")
    _touch(tmp_path / "ok.mp4")
    r = discover_videos([tmp_path / "pic.jpg", tmp_path / "doc.pdf", tmp_path / "ok.mp4"])
    assert r.added == 1
    assert r.ignored == 2


def test_duplicate_dedup(tmp_path: Path):
    f = _touch(tmp_path / "a.mp4")
    r = discover_videos([f, f])
    assert r.added == 1
    assert r.duplicates_skipped == 1


def test_windows_path_case_dedup(tmp_path: Path):
    f = _touch(tmp_path / "Video.MP4")
    with patch("framesentry.core.input_discovery.sys.platform", "win32"):
        with patch("framesentry.core.input_discovery.os.path.normcase", side_effect=lambda s: s.lower()):
            k1 = normalize_video_key(str(tmp_path / "Video.MP4"))
            k2 = normalize_video_key(str(tmp_path / "video.mp4"))
            assert k1 == k2
            assert k1 == k1.casefold()

    # Existing-set dedup: same file via different path objects
    r1 = discover_videos([f])
    r2 = discover_videos([f], existing={normalize_video_key(f)})
    assert r2.added == 0
    assert r2.duplicates_skipped == 1


def test_add_button_and_drag_same_api_no_dup(tmp_path: Path):
    """Buttons and DnD must share discover_videos — second call dedups."""
    f = _touch(tmp_path / "clip.mp4")
    # Simulate Add Files
    r1 = discover_videos([str(f)])
    existing = {normalize_video_key(v) for v in r1.videos}
    # Simulate Drop of same file
    r2 = discover_videos([str(f)], existing=existing)
    assert r1.added == 1
    assert r2.added == 0
    assert r2.duplicates_skipped == 1


def test_invalid_path_does_not_abort_batch(tmp_path: Path):
    good = _touch(tmp_path / "good.mp4")
    r = discover_videos(
        [
            str(tmp_path / "missing.mp4"),
            str(good),
            str(tmp_path / "nope.jpg"),
            "/this/path/does/not/exist/video.mkv",
        ]
    )
    assert r.added == 1
    assert r.ignored >= 2
    assert Path(r.videos[0]).name == "good.mp4"


def test_toast_message():
    r = DiscoveryResult(added=12, ignored=4, duplicates_skipped=1)
    msg = r.toast_message
    assert "已添加 12 个视频" in msg
    assert "忽略 4 个不支持项目" in msg


def test_is_supported_video():
    assert is_supported_video("x.mp4")
    assert is_supported_video("x.MKV")
    assert not is_supported_video("x.txt")
