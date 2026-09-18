"""Shared data types for scanning and review."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class VideoStatus(str, Enum):
    WAITING = "WAITING"
    SCANNING = "SCANNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


@dataclass(slots=True)
class DetectionHit:
    """One raw detection on a single sampled frame."""

    frame_index: int
    timestamp_sec: float
    timestamp_str: str
    class_name: str
    score: float
    box: list[int]  # [x, y, w, h]
    frame_filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectionHit:
        return cls(
            frame_index=int(data["frame_index"]),
            timestamp_sec=float(data["timestamp_sec"]),
            timestamp_str=str(data["timestamp_str"]),
            class_name=str(data["class_name"]),
            score=float(data["score"]),
            box=list(data["box"]),
            frame_filename=data.get("frame_filename"),
        )


@dataclass(slots=True)
class DetectionEvent:
    """Cluster of adjacent hits (same/compatible class within merge window)."""

    start_sec: float
    end_sec: float
    start_str: str
    end_str: str
    count: int
    classes: list[str]
    max_score: float
    best_frame_index: int
    best_frame_filename: str | None
    best_timestamp_sec: float
    hit_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectionEvent:
        return cls(
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            start_str=str(data["start_str"]),
            end_str=str(data["end_str"]),
            count=int(data["count"]),
            classes=list(data["classes"]),
            max_score=float(data["max_score"]),
            best_frame_index=int(data["best_frame_index"]),
            best_frame_filename=data.get("best_frame_filename"),
            best_timestamp_sec=float(data["best_timestamp_sec"]),
            hit_indices=list(data.get("hit_indices", [])),
        )


@dataclass
class VideoJob:
    path: str
    name: str
    status: VideoStatus = VideoStatus.WAITING
    progress: float = 0.0
    hit_count: int = 0
    error: str = ""
    output_dir: str = ""
