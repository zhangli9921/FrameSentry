"""Output directory layout."""

from __future__ import annotations

import hashlib
from pathlib import Path

from framesentry.core.config import OUTPUT_DIR_SUFFIX

# Stable short id length (8–12 hex chars from SHA-256).
_PATH_ID_HEX_LEN = 10


class UnsafeReviewPathError(RuntimeError):
    """Raised when a review path is unsafe to clear (e.g. review_dir is a symlink)."""


def normalize_abs_path_for_id(video_path: str | Path) -> str:
    """Case-normalized absolute path for stable review-dir ids (Windows-safe).

    Uses resolve() when possible; falls back to absolute(). Replaces backslashes,
    then casefold() so the same Windows path with different casing maps to one id.
    Does not use builtin hash().
    """
    p = Path(video_path)
    try:
        abs_p = p.resolve()
    except OSError:
        abs_p = p.absolute()
    # Normalize separators then casefold for case-insensitive platforms.
    return str(abs_p).replace("\\", "/").casefold()


def path_stable_id(video_path: str | Path, hex_len: int = _PATH_ID_HEX_LEN) -> str:
    """SHA-256 hex prefix of case-normalized absolute path (not builtin hash)."""
    if hex_len < 8 or hex_len > 12:
        raise ValueError("hex_len must be between 8 and 12")
    digest = hashlib.sha256(normalize_abs_path_for_id(video_path).encode("utf-8"))
    return digest.hexdigest()[:hex_len]


def review_dir_for_video(video_path: str | Path, output_root: str | Path) -> Path:
    """``<output_root>/<stem>.<path_id>.framesentry_review``.

    Same absolute path (case-insensitive) → same dir; different paths with the
    same stem → different dirs. Never pollutes the source directory by default.
    """
    stem = Path(video_path).stem
    short_id = path_stable_id(video_path)
    return Path(output_root) / f"{stem}.{short_id}{OUTPUT_DIR_SUFFIX}"


def ensure_review_dirs(review_dir: str | Path) -> tuple[Path, Path]:
    """Create review dir and frames/ subdir. Returns (review_dir, frames_dir)."""
    review = Path(review_dir)
    frames = review / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    return review, frames


def clear_review_dir(review_dir: str | Path) -> Path:
    """Safely clear one video's review dir (frames + results.json) for rescan.

    Only deletes content under ``review_dir`` itself. Does not wipe the output
    root, does not delete the source video, and does not follow symlinks that
    point outside the review directory.

    If ``review_dir`` itself already exists and is a symlink, refuse cleanup
    and raise ``UnsafeReviewPathError`` (do not follow / resolve into the
    target or delete external contents). If only ``frames/`` is a symlink,
    the link node may be removed without following into the target.
    """
    review = Path(review_dir)
    if not review.exists():
        review.mkdir(parents=True, exist_ok=True)
        (review / "frames").mkdir(parents=True, exist_ok=True)
        return review

    # review_dir itself must not be a symlink — resolve() would follow it and
    # subsequent deletes could wipe an out-of-tree target.
    if review.is_symlink():
        raise UnsafeReviewPathError(
            f"Refusing to clear review directory that is a symlink: {review} "
            f"(target would be followed by resolve(); out-of-tree delete blocked). "
            "Remove the symlink manually if this path is intentional."
        )

    try:
        review_resolved = review.resolve()
    except OSError:
        review_resolved = review.absolute()

    frames = review / "frames"
    if frames.is_dir() and not frames.is_symlink():
        for child in list(frames.iterdir()):
            _safe_unlink_under(child, review_resolved)
    elif frames.is_symlink():
        # Do not follow a frames symlink outside; remove the link only if under review.
        _safe_unlink_under(frames, review_resolved)

    results = review / "results.json"
    if results.exists():
        _safe_unlink_under(results, review_resolved)

    # Recreate empty frames/ for the upcoming scan.
    frames_dir = review / "frames"
    if not frames_dir.exists():
        frames_dir.mkdir(parents=True, exist_ok=True)
    return review


def _safe_unlink_under(path: Path, root_resolved: Path) -> None:
    """Delete file/dir only if its real path stays under root_resolved."""
    try:
        if path.is_symlink():
            # Remove the symlink node itself; do not recurse into the target.
            real_parent = path.parent.resolve()
            if root_resolved not in real_parent.parents and real_parent != root_resolved:
                return
            path.unlink(missing_ok=True)
            return
        resolved = path.resolve()
    except OSError:
        return

    if root_resolved not in resolved.parents and resolved != root_resolved:
        return

    if path.is_dir() and not path.is_symlink():
        # Delete children first, still checking containment.
        for child in list(path.iterdir()):
            _safe_unlink_under(child, root_resolved)
        try:
            path.rmdir()
        except OSError:
            pass
    else:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
