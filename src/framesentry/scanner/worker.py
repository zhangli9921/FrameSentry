"""Video scanning off the UI thread (callable from QThread)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2

from framesentry.core.config import (
    DEFAULT_MERGE_WINDOW_SEC,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_THRESHOLD,
    TARGET_CLASSES,
)
from framesentry.core.events import cluster_hits
from framesentry.core.logging_setup import get_logger
from framesentry.core.timecode import format_timestamp, frame_step_from_fps
from framesentry.core.types import DetectionHit
from framesentry.core.validation import filter_target_detections, validate_sample_fps, validate_threshold
from framesentry.storage.frames import annotate_and_save_frame, safe_frame_filename
from framesentry.storage.paths import ensure_review_dirs, review_dir_for_video
from framesentry.storage.results import save_results

logger = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


class ScanCancelled(Exception):
    """Raised when the user cancels the current scan."""


@dataclass
class ScanSettings:
    sample_fps: float = DEFAULT_SAMPLE_FPS
    threshold: float = DEFAULT_THRESHOLD
    merge_window_sec: float = DEFAULT_MERGE_WINDOW_SEC
    device: str = "cpu"
    output_root: str = ""
    model_path: str | None = None
    save_frames: bool = True


def scan_video(
    video_path: str,
    detector: Any,
    settings: ScanSettings,
    *,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict[str, Any]:
    """Scan one video. One failure must be handled by the caller queue (this raises).

    Decode errors: skip bad frames when possible; mark warning if unreadable.
    """
    validate_sample_fps(settings.sample_fps)
    validate_threshold(settings.threshold)

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    output_root = settings.output_root or str(path.parent)
    review_dir = review_dir_for_video(path, output_root)
    review_dir, frames_dir = ensure_review_dirs(review_dir)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if video_fps <= 0:
        video_fps = 25.0
        logger.warning("invalid video FPS for %s; assuming 25", video_path)

    step = frame_step_from_fps(video_fps, settings.sample_fps)
    hits: list[DetectionHit] = []
    warnings: list[str] = []
    sampled = 0
    decode_failures = 0
    frame_idx = 0

    def progress(pct: float, msg: str) -> None:
        if on_progress:
            on_progress(pct, msg)

    try:
        while True:
            if should_cancel and should_cancel():
                raise ScanCancelled(f"cancelled: {video_path}")

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            if frame is None or getattr(frame, "size", 0) == 0:
                decode_failures += 1
                warnings.append(f"bad frame at index {frame_idx}")
                frame_idx += 1
                continue

            timestamp_sec = frame_idx / video_fps
            ts_str = format_timestamp(timestamp_sec)
            try:
                raw = detector.detect(frame)
            except Exception as exc:  # noqa: BLE001 — keep queue alive per-frame
                decode_failures += 1
                warnings.append(f"detect error at frame {frame_idx}: {exc}")
                frame_idx += 1
                continue

            filtered = filter_target_detections(
                raw, threshold=settings.threshold, target_classes=TARGET_CLASSES
            )
            if filtered:
                # One annotated image per sampled frame that has hits (primary class = best score)
                best = max(filtered, key=lambda d: float(d.get("score", 0.0)))
                fname = safe_frame_filename(
                    frame_idx,
                    ts_str,
                    str(best.get("class")),
                    float(best.get("score", 0.0)),
                )
                if settings.save_frames:
                    try:
                        annotate_and_save_frame(frame, filtered, frames_dir / fname)
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"frame save error {frame_idx}: {exc}")
                        fname = None  # type: ignore[assignment]
                for det in filtered:
                    hits.append(
                        DetectionHit(
                            frame_index=frame_idx,
                            timestamp_sec=timestamp_sec,
                            timestamp_str=ts_str,
                            class_name=str(det.get("class")),
                            score=float(det.get("score", 0.0)),
                            box=list(det.get("box") or [0, 0, 0, 0]),
                            frame_filename=fname if settings.save_frames else None,
                        )
                    )

            sampled += 1
            if frame_count > 0:
                pct = min(99.0, 100.0 * (frame_idx + 1) / frame_count)
            else:
                pct = 0.0
            if sampled % 5 == 0:
                progress(pct, f"frame {frame_idx} hits={len(hits)}")

            frame_idx += 1
    finally:
        cap.release()

    events = cluster_hits(hits, merge_window_sec=settings.merge_window_sec)
    meta = {
        "sample_fps": settings.sample_fps,
        "threshold": settings.threshold,
        "merge_window_sec": settings.merge_window_sec,
        "device": settings.device,
        "video_fps": video_fps,
        "frame_step": step,
        "sampled_frames": sampled,
        "decode_failures": decode_failures,
        "warnings": warnings,
        "target_classes": sorted(TARGET_CLASSES),
    }
    save_results(
        review_dir,
        video_path=str(path.resolve()),
        hits=hits,
        events=events,
        meta=meta,
    )
    progress(100.0, f"done hits={len(hits)} events={len(events)}")

    status_note = ""
    if frame_count == 0 and sampled == 0:
        status_note = "unreadable or empty video"
    elif decode_failures and not hits and sampled == 0:
        status_note = "decode failures; no usable frames"

    return {
        "review_dir": str(review_dir),
        "hits": hits,
        "events": events,
        "hit_count": len(hits),
        "event_count": len(events),
        "warnings": warnings,
        "status_note": status_note,
    }
