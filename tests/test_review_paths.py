"""Review dir path id: same path → same dir; different paths / stems → distinct."""

from __future__ import annotations

import hashlib
from pathlib import Path

from framesentry.storage.paths import (
    clear_review_dir,
    normalize_abs_path_for_id,
    path_stable_id,
    review_dir_for_video,
)


def test_same_path_same_review_dir(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "out"
    a = review_dir_for_video(video, out)
    b = review_dir_for_video(video, out)
    assert a == b
    assert video.stem in a.name
    assert a.name.endswith(".framesentry_review")
    parts = a.name[: -len(".framesentry_review")].split(".")
    # stem may contain dots; last segment is the hex id
    hex_id = parts[-1]
    assert 8 <= len(hex_id) <= 12
    assert all(c in "0123456789abcdef" for c in hex_id)


def test_same_stem_different_paths_different_dirs(tmp_path: Path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    v1 = d1 / "clip.mp4"
    v2 = d2 / "clip.mp4"
    v1.write_bytes(b"1")
    v2.write_bytes(b"2")
    out = tmp_path / "out"
    r1 = review_dir_for_video(v1, out)
    r2 = review_dir_for_video(v2, out)
    assert r1 != r2
    assert r1.name.startswith("clip.")
    assert r2.name.startswith("clip.")


def test_casefold_normalize_stable():
    p = Path("/Videos/Clip.MP4")
    n = normalize_abs_path_for_id(p)
    assert n == n.casefold()
    expected = hashlib.sha256(n.encode("utf-8")).hexdigest()[:10]
    assert path_stable_id(p) == expected


def test_path_stable_id_not_builtin_hash(tmp_path: Path):
    video = tmp_path / "x.mp4"
    video.write_bytes(b"z")
    sid = path_stable_id(video)
    assert sid != str(hash(str(video.resolve())))
    assert len(sid) == 10
    # hashlib.sha256, not builtin hash()
    import inspect

    src = inspect.getsource(path_stable_id)
    assert "sha256" in src
    assert "hash(" not in src.replace("sha256", "")


def test_clear_review_dir_removes_stale_frames(tmp_path: Path):
    review = tmp_path / "clip.abc123def0.framesentry_review"
    frames = review / "frames"
    frames.mkdir(parents=True)
    stale = frames / "old.jpg"
    stale.write_bytes(b"jpeg")
    results = review / "results.json"
    results.write_text("{}", encoding="utf-8")
    outside = tmp_path / "source.mp4"
    outside.write_bytes(b"video")

    clear_review_dir(review)

    assert not stale.exists()
    assert not results.exists()
    assert (review / "frames").is_dir()
    assert outside.exists()
    assert tmp_path.exists()


def test_clear_review_dir_refuses_symlink_review_root(tmp_path: Path):
    """If review_dir itself is a symlink, refuse cleanup — do not wipe target."""
    import os
    import pytest

    from framesentry.storage.paths import UnsafeReviewPathError

    external = tmp_path / "external_real"
    frames = external / "frames"
    frames.mkdir(parents=True)
    keep = frames / "keep.jpg"
    keep.write_bytes(b"jpeg-keep")
    ext_results = external / "results.json"
    ext_results.write_text('{"keep": true}', encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()
    link = out / "clip.abc123def0.framesentry_review"

    try:
        os.symlink(external, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create directory symlink on this platform: {exc}")

    assert link.is_symlink()

    with pytest.raises(UnsafeReviewPathError):
        clear_review_dir(link)

    assert keep.is_file()
    assert keep.read_bytes() == b"jpeg-keep"
    assert ext_results.is_file()
    assert '"keep": true' in ext_results.read_text(encoding="utf-8")
    # Symlink itself must still exist (no unlink-and-continue / auto-replace).
    assert link.is_symlink()
