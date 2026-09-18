"""Application defaults and constants."""

from __future__ import annotations

from typing import Final

# Recall-first defaults — do not raise without deliberate product decision.
DEFAULT_SAMPLE_FPS: Final[float] = 2.0
DEFAULT_THRESHOLD: Final[float] = 0.35
DEFAULT_MERGE_WINDOW_SEC: Final[float] = 2.0

SAMPLE_FPS_PRESETS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0)
SAMPLE_FPS_MIN: Final[float] = 0.1
SAMPLE_FPS_MAX: Final[float] = 30.0

THRESHOLD_MIN: Final[float] = 0.0
THRESHOLD_MAX: Final[float] = 1.0

VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".mp4", ".flv", ".mkv", ".mov", ".avi", ".webm"}
)

# Actual NudeNet 3.4.2 label names (not aliases).
TARGET_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "BUTTOCKS_EXPOSED",
    }
)

OUTPUT_DIR_SUFFIX: Final[str] = ".framesentry_review"
