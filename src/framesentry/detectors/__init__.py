"""Detector backends. GUI must not import nudenet directly — use factory helpers."""

from framesentry.detectors.base import DetectorBackend, OrtProviderInfo
from framesentry.detectors.nudenet_backend import (
    NudeNetBackend,
    create_nudenet_backend,
    get_ort_provider_info,
)

__all__ = [
    "DetectorBackend",
    "OrtProviderInfo",
    "NudeNetBackend",
    "create_nudenet_backend",
    "get_ort_provider_info",
]
