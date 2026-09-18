"""Video scanning off the UI thread (callable from QThread)."""

from __future__ import annotations

import time
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
from framesentry.storage.paths import clear_review_dir, ensure_review_dirs, review_dir_for_video
from framesentry.storage.results import save_results

logger = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


class ScanCancelled(Exception):
    """Raised when the user cancels the current scan."""


class UnreadableVideoError(RuntimeError):
    """Raised when no usable frames could be sampled (must surface as FAILED)."""


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

    Decode errors are counted separately from detector inference errors.
    If final ``sampled_frames == 0``, raises ``UnreadableVideoError`` (FAILED).
    Partial early EOF vs declared ``frame_count`` logs a warning but may COMPLETE.
    """
    validate_sample_fps(settings.sample_fps)
    validate_threshold(settings.threshold)

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    output_root = settings.output_root or str(path.parent)
    review_dir = review_dir_for_video(path, output_root)
    # Rescan: clear stale frames/results for THIS review dir only.
    clear_review_dir(review_dir)
    review_dir, frames_dir = ensure_review_dirs(review_dir)

    t0 = time.perf_counter()
    logger.info(
        "scan start path=%s sample_fps=%s threshold=%s device=%s review_dir=%s",
        path,
        settings.sample_fps,
        settings.threshold,
        settings.device,
        review_dir,
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if video_fps <= 0:
        video_fps = 25.0
        logger.warning("invalid video FPS for %s; assuming 25", video_path)

    step = frame_step_from_fps(video_fps, settings.sample_fps)
    logger.info(
        "video_fps=%s frame_count=%s frame_step=%s",
        video_fps,
        frame_count,
        step,
    )

    hits: list[DetectionHit] = []
    warnings: list[str] = []
    sampled = 0
    decode_failures = 0
    detect_errors = 0
    frame_idx = 0
    last_read_ok_idx = -1

    def progress(pct: float, msg: str) -> None:
        if on_progress:
            on_progress(pct, msg)

    try:
        while True:
            if should_cancel and should_cancel():
                logger.info("scan CANCELLED path=%s", path)
                raise ScanCancelled(f"cancelled: {video_path}")

            ret, frame = cap.read()
            if not ret:
                break

            last_read_ok_idx = frame_idx

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            if frame is None or getattr(frame, "size", 0) == 0:
                decode_failures += 1
                warnings.append(f"bad frame at index {frame_idx}")
                logger.warning("decode warning: bad frame at index %s", frame_idx)
                frame_idx += 1
                continue

            timestamp_sec = frame_idx / video_fps
            ts_str = format_timestamp(timestamp_sec)
            try:
                raw = detector.detect(frame)
            except Exception as exc:  # noqa: BLE001 — keep queue alive per-frame
                detect_errors += 1
                warnings.append(f"detect error at frame {frame_idx}: {exc}")
                logger.warning("detect error at frame %s: %s", frame_idx, exc)
                frame_idx += 1
                continue

            filtered = filter_target_detections(
                raw, threshold=settings.threshold, target_classes=TARGET_CLASSES
            )
            if filtered:
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

    # Early EOF vs declared frame_count → warning; may still COMPLETE if sampled > 0.
    if frame_count > 0 and last_read_ok_idx >= 0:
        expected_last = frame_count - 1
        if last_read_ok_idx < expected_last - 1:
            msg = (
                f"decode ended early: last readable index {last_read_ok_idx} "
                f"vs declared frame_count {frame_count}"
            )
            warnings.append(msg)
            logger.warning(msg)

    duration = time.perf_counter() - t0

    if sampled == 0:
        logger.error(
            "scan FAILED unreadable path=%s decode_failures=%s detect_errors=%s "
            "duration=%.3fs",
            path,
            decode_failures,
            detect_errors,
            duration,
        )
        raise UnreadableVideoError(
            f"unreadable video (0 usable sampled frames): {video_path} "
            f"(decode_failures={decode_failures}, detect_errors={detect_errors})"
        )

    events = cluster_hits(hits, merge_window_sec=settings.merge_window_sec)
    meta = {
        "sample_fps": settings.sample_fps,
        "threshold": settings.threshold,
        "merge_window_sec": settings.merge_window_sec,
        "device": settings.device,
        "video_fps": video_fps,
        "frame_step": step,
        "frame_count_declared": frame_count,
        "sampled_frames": sampled,
        "decode_failures": decode_failures,
        "detect_errors": detect_errors,
        "warnings": warnings,
        "target_classes": sorted(TARGET_CLASSES),
        "duration_sec": round(duration, 3),
    }
    save_results(
        review_dir,
        video_path=str(path.resolve()),
        hits=hits,
        events=events,
        meta=meta,
    )
    progress(100.0, f"done hits={len(hits)} events={len(events)}")

    logger.info(
        "scan end path=%s sampled_frames=%s hit_count=%s event_count=%s "
        "decode_failures=%s detect_errors=%s duration=%.3fs",
        path,
        sampled,
        len(hits),
        len(events),
        decode_failures,
        detect_errors,
        duration,
    )

    return {
        "review_dir": str(review_dir),
        "hits": hits,
        "events": events,
        "hit_count": len(hits),
        "event_count": len(events),
        "sampled_frames": sampled,
        "decode_failures": decode_failures,
        "detect_errors": detect_errors,
        "warnings": warnings,
        "status_note": "",
    }
