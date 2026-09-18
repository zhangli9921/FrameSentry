"""Detector backend protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class OrtProviderInfo:
    """ONNX Runtime provider availability (honest — never claim CUDA if absent)."""

    available_providers: tuple[str, ...]
    cuda_listed: bool
    gpu_mode_usable: bool  # True only if CUDAExecutionProvider is listed

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_providers": list(self.available_providers),
            "cuda_listed": self.cuda_listed,
            "gpu_mode_usable": self.gpu_mode_usable,
        }


@runtime_checkable
class DetectorBackend(Protocol):
    """Minimal detector interface used by the scanner."""

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Detect on a single BGR ndarray. Returns list of {class, score, box}."""
        ...

    def detect_batch(self, images_bgr: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        """Detect on a list of BGR ndarrays. May fall back to per-frame detect."""
        ...
