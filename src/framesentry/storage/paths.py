"""Output directory layout."""

from __future__ import annotations

from pathlib import Path

from framesentry.core.config import OUTPUT_DIR_SUFFIX


def review_dir_for_video(video_path: str | Path, output_root: str | Path) -> Path:
    """``<output_root>/<stem>.framesentry_review`` — never pollutes source dir by default."""
    stem = Path(video_path).stem
    return Path(output_root) / f"{stem}{OUTPUT_DIR_SUFFIX}"


def ensure_review_dirs(review_dir: str | Path) -> tuple[Path, Path]:
    """Create review dir and frames/ subdir. Returns (review_dir, frames_dir)."""
    review = Path(review_dir)
    frames = review / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    return review, frames
