"""Input validation helpers."""

from __future__ import annotations

from framesentry.core.config import (
    SAMPLE_FPS_MAX,
    SAMPLE_FPS_MIN,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
)


def validate_threshold(value: float) -> float:
    """Validate and return threshold in [0.0, 1.0]. Raises ValueError otherwise."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold must be a number, got {value!r}") from exc
    if v < THRESHOLD_MIN or v > THRESHOLD_MAX:
        raise ValueError(
            f"threshold must be in [{THRESHOLD_MIN}, {THRESHOLD_MAX}], got {v}"
        )
    return v


def validate_sample_fps(value: float) -> float:
    """Validate sample FPS in a reasonable range."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_fps must be a number, got {value!r}") from exc
    if v < SAMPLE_FPS_MIN or v > SAMPLE_FPS_MAX:
        raise ValueError(
            f"sample_fps must be in [{SAMPLE_FPS_MIN}, {SAMPLE_FPS_MAX}], got {v}"
        )
    return v


def filter_target_detections(
    detections: list[dict],
    *,
    threshold: float,
    target_classes: set[str] | frozenset[str],
) -> list[dict]:
    """Keep detections whose class is in target set and score >= threshold."""
    thr = validate_threshold(threshold)
    out: list[dict] = []
    for det in detections:
        cls = det.get("class") or det.get("class_name")
        score = float(det.get("score", 0.0))
        if cls in target_classes and score >= thr:
            out.append(det)
    return out
