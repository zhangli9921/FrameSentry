"""Timecode helpers."""

from __future__ import annotations

import math


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH-MM-SS.mmm (filesystem-safe, no colons)."""
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{ms:03d}"


def frame_step_from_fps(video_fps: float, sample_fps: float) -> int:
    """Return integer frame stride for sampling at sample_fps.

    Guarantees at least 1. If sample_fps >= video_fps, returns 1 (every frame).
    """
    if video_fps <= 0 or sample_fps <= 0:
        return 1
    if sample_fps >= video_fps:
        return 1
    step = int(round(video_fps / sample_fps))
    return max(1, step)
