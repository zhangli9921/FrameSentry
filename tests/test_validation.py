"""Threshold and class filter validation."""

import pytest

from framesentry.core.config import TARGET_CLASSES
from framesentry.core.validation import (
    filter_target_detections,
    validate_sample_fps,
    validate_threshold,
)


def test_threshold_ok():
    assert validate_threshold(0.35) == 0.35
    assert validate_threshold(0.0) == 0.0
    assert validate_threshold(1.0) == 1.0


def test_threshold_bad():
    with pytest.raises(ValueError):
        validate_threshold(-0.1)
    with pytest.raises(ValueError):
        validate_threshold(1.1)
    with pytest.raises(ValueError):
        validate_threshold("x")


def test_sample_fps():
    assert validate_sample_fps(2.0) == 2.0
    with pytest.raises(ValueError):
        validate_sample_fps(0)
    with pytest.raises(ValueError):
        validate_sample_fps(100)


def test_class_filter():
    dets = [
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.55, "box": [1, 2, 3, 4]},
        {"class": "FACE_FEMALE", "score": 0.99, "box": [1, 2, 3, 4]},
        {"class": "BUTTOCKS_EXPOSED", "score": 0.20, "box": [1, 2, 3, 4]},
        {"class": "ANUS_EXPOSED", "score": 0.40, "box": [1, 2, 3, 4]},
    ]
    out = filter_target_detections(dets, threshold=0.35, target_classes=TARGET_CLASSES)
    classes = {d["class"] for d in out}
    assert classes == {"FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED"}
    assert "FACE_FEMALE" not in classes
