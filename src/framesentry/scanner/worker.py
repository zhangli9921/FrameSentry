"""Video scanning off the UI thread (callable from QThread)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from framesentry.core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CPU_BATCH_SIZE,
    DEFAULT_MERGE_WINDOW_SEC,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_THRESHOLD,
    TARGET_CLASSES,
)
from framesentry.core.events import cluster_hits
from framesentry.core.logging_setup import get_logger
from framesentry.core.timecode import format_timestamp, frame_step_from_fps
from framesentry.core.types import DetectionHit
from framesentry.core.validation import (
    filter_target_detections,
    validate_sample_fps,
    validate_threshold,
)
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
    batch_size: int = DEFAULT_BATCH_SIZE
    intermediate_dir: str = ""


@dataclass
class _FrameSample:
    frame_idx: int
    timestamp_sec: float
    timestamp_str: str
    frame: np.ndarray


def _effective_batch_size(settings: ScanSettings, detector: Any) -> tuple[int, Any, bool]:
    """Return (batch_size, onnx_batch_dim, supports_batch_gt1)."""
    batch_size = max(1, int(settings.batch_size or 1))
    device = (settings.device or "cpu").strip().lower()
    if device == "cpu":
        batch_size = min(batch_size, DEFAULT_CPU_BATCH_SIZE)

    onnx_batch_dim: Any = None
    supports = True
    if hasattr(detector, "onnx_input_batch_dim"):
        try:
            onnx_batch_dim = detector.onnx_input_batch_dim()
        except Exception:  # noqa: BLE001
            onnx_batch_dim = None
    if hasattr(detector, "supports_batch_gt1"):
        try:
            supports = bool(detector.supports_batch_gt1())
        except Exception:  # noqa: BLE001
            supports = True
    if not supports and batch_size > 1:
        logger.warning(
            "ONNX input batch dim does not support >1; forcing batch_size=1 (no fake-batch)"
        )
        batch_size = 1
    return batch_size, onnx_batch_dim, supports


def scan_video(
    video_path: str,
    detector: Any,
    settings: ScanSettings,
    *,
    scan_input_path: str | None = None,
    preprocess_meta: dict[str, Any] | None = None,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict[str, Any]:
    """Scan one video. Caller queue must handle failures (this raises).

    ``video_path`` is the **original source** (review id / results.json / HTML).
    When ``scan_input_path`` is set, frames are decoded from that intermediate MP4 only.
    """
    validate_sample_fps(settings.sample_fps)
    validate_threshold(settings.threshold)

    source_path = Path(video_path)
    input_path = Path(scan_input_path) if scan_input_path else source_path
    if not input_path.is_file():
        raise FileNotFoundError(f"scan input not found: {input_path}")

    output_root = settings.output_root or str(source_path.parent)
    # Review dir SHA is based on **original source path**, not intermediate.
    review_dir = review_dir_for_video(source_path, output_root)
    clear_review_dir(review_dir)
    review_dir, frames_dir = ensure_review_dirs(review_dir)

    batch_size, onnx_batch_dim, supports_batch = _effective_batch_size(settings, detector)

    active_providers: list[str] = []
    if hasattr(detector, "session") and getattr(detector, "session", None) is not None:
        try:
            active_providers = list(detector.session.get_providers())
        except Exception:  # noqa: BLE001
            active_providers = []

    t0 = time.perf_counter()
    logger.info(
        "scan start source=%s scan_input=%s sample_fps=%s threshold=%s device=%s "
        "batch_size=%s onnx_batch_dim=%s review_dir=%s",
        source_path,
        input_path,
        settings.sample_fps,
        settings.threshold,
        settings.device,
        batch_size,
        onnx_batch_dim,
        review_dir,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if video_fps <= 0:
        video_fps = 25.0
        logger.warning("invalid video FPS for %s; assuming 25", input_path)

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
    decoded_frames = 0
    batch_runs = 0

    t_decode = 0.0
    t_preprocess_tensor = 0.0
    t_inference = 0.0
    t_frame_save = 0.0
    t_postprocess = 0.0

    buffer: list[_FrameSample] = []

    def progress(pct: float, msg: str) -> None:
        if on_progress:
            on_progress(pct, msg)

    def progress_update(extra: str = "") -> None:
        ts = (
            buffer[-1].timestamp_str
            if buffer
            else (
                format_timestamp(last_read_ok_idx / video_fps)
                if last_read_ok_idx >= 0
                else "00:00:00"
            )
        )
        if frame_count > 0:
            pct = min(99.0, 100.0 * max(frame_idx, 1) / frame_count)
            msg = f"frame {frame_idx} hits={len(hits)} sampled={sampled}"
        else:
            # Indeterminate: never stuck at bogus 0%.
            pct = -1.0
            msg = f"sampled={sampled} t={ts} hits={len(hits)}"
        if extra:
            msg = f"{msg} {extra}"
        progress(pct, msg)

    def flush_batch() -> None:
        nonlocal sampled, detect_errors, batch_runs, t_preprocess_tensor, t_inference, t_frame_save, t_postprocess
        if not buffer:
            return
        if should_cancel and should_cancel():
            raise ScanCancelled(f"cancelled: {video_path}")

        frames = [s.frame for s in buffer]
        frame_lo = buffer[0].frame_idx
        frame_hi = buffer[-1].frame_idx
        try:
            if hasattr(detector, "detect_batch"):
                batch_out = detector.detect_batch(frames, batch_size=len(frames))
            else:
                batch_out = [detector.detect(f) for f in frames]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "detect_batch failed frames [%s, %s] (n=%s): %s",
                frame_lo,
                frame_hi,
                len(buffer),
                exc,
            )
            buffer.clear()
            raise

        timing = getattr(detector, "last_batch_timing", None) or {}
        t_preprocess_tensor += float(timing.get("preprocess_tensor_sec", 0.0))
        t_inference += float(timing.get("inference_sec", 0.0))
        # NudeNet _postprocess time; target-class filter measured below.
        t_postprocess += float(timing.get("postprocess_sec", 0.0))
        batch_runs += 1

        for sample, raw in zip(buffer, batch_out):
            t_filt0 = time.perf_counter()
            try:
                filtered = filter_target_detections(
                    raw, threshold=settings.threshold, target_classes=TARGET_CLASSES
                )
            except Exception as exc:  # noqa: BLE001
                detect_errors += 1
                warnings.append(f"filter error at frame {sample.frame_idx}: {exc}")
                sampled += 1
                t_postprocess += time.perf_counter() - t_filt0
                continue
            t_postprocess += time.perf_counter() - t_filt0

            fname: str | None = None
            if filtered:
                best = max(filtered, key=lambda d: float(d.get("score", 0.0)))
                fname = safe_frame_filename(
                    sample.frame_idx,
                    sample.timestamp_str,
                    str(best.get("class")),
                    float(best.get("score", 0.0)),
                )
                if settings.save_frames:
                    t_fs0 = time.perf_counter()
                    try:
                        annotate_and_save_frame(sample.frame, filtered, frames_dir / fname)
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"frame save error {sample.frame_idx}: {exc}")
                        fname = None
                    t_frame_save += time.perf_counter() - t_fs0
                for det in filtered:
                    hits.append(
                        DetectionHit(
                            frame_index=sample.frame_idx,
                            timestamp_sec=sample.timestamp_sec,
                            timestamp_str=sample.timestamp_str,
                            class_name=str(det.get("class")),
                            score=float(det.get("score", 0.0)),
                            box=list(det.get("box") or [0, 0, 0, 0]),
                            frame_filename=fname if settings.save_frames else None,
                        )
                    )
            sampled += 1

        progress_update(f"batches={batch_runs}")
        buffer.clear()

    try:
        while True:
            if should_cancel and should_cancel():
                logger.info("scan CANCELLED path=%s", source_path)
                raise ScanCancelled(f"cancelled: {video_path}")

            t_d0 = time.perf_counter()
            ret, frame = cap.read()
            t_decode += time.perf_counter() - t_d0
            if not ret:
                break

            last_read_ok_idx = frame_idx
            decoded_frames += 1

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
            buffer.append(
                _FrameSample(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    timestamp_str=ts_str,
                    frame=frame,
                )
            )
            if len(buffer) >= batch_size:
                flush_batch()

            frame_idx += 1

        flush_batch()
    finally:
        cap.release()

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
            source_path,
            decode_failures,
            detect_errors,
            duration,
        )
        raise UnreadableVideoError(
            f"unreadable video (0 usable sampled frames): {video_path} "
            f"(decode_failures={decode_failures}, detect_errors={detect_errors})"
        )

    events = cluster_hits(hits, merge_window_sec=settings.merge_window_sec)

    sampled_fps = (sampled / duration) if duration > 0 else 0.0
    ms_per_batch = (t_inference * 1000.0 / batch_runs) if batch_runs else 0.0
    ms_per_frame = (t_inference * 1000.0 / sampled) if sampled else 0.0

    prep = preprocess_meta or {}
    meta: dict[str, Any] = {
        "sample_fps": settings.sample_fps,
        "threshold": settings.threshold,
        "merge_window_sec": settings.merge_window_sec,
        "device": settings.device,
        "video_fps": video_fps,
        "frame_step": step,
        "frame_count_declared": frame_count,
        "sampled_frames": sampled,
        "decoded_frames": decoded_frames,
        "decode_failures": decode_failures,
        "detect_errors": detect_errors,
        "warnings": warnings,
        "target_classes": sorted(TARGET_CLASSES),
        "duration_sec": round(duration, 3),
        "source_video_path": str(prep.get("source_video_path") or source_path),
        "scan_input_path": str(prep.get("scan_input_path") or input_path),
        "intermediate_path": prep.get("intermediate_path"),
        "preprocess_duration_sec": prep.get("preprocess_duration_sec"),
        "ffmpeg_exit_code": prep.get("ffmpeg_exit_code"),
        "batch_size": batch_size,
        "batch_runs": batch_runs,
        "onnx_input_batch_dim": onnx_batch_dim,
        "onnx_supports_batch_gt1": supports_batch,
        "active_ort_providers": active_providers,
        "decode_sec": round(t_decode, 3),
        "preprocess_tensor_sec": round(t_preprocess_tensor, 3),
        "inference_sec": round(t_inference, 3),
        "postprocess_sec": round(t_postprocess, 3),
        "frame_save_sec": round(t_frame_save, 3),
        "total_sec": round(duration, 3),
        "sampled_fps": round(sampled_fps, 3),
        "ms_per_batch": round(ms_per_batch, 3),
        "ms_per_frame": round(ms_per_frame, 3),
        "inference_ms_per_frame": round(
            (t_inference * 1000.0 / sampled) if sampled else 0.0, 3
        ),
    }

    resolved_source = str(source_path.resolve()) if source_path.exists() else str(source_path)
    save_results(
        review_dir,
        video_path=resolved_source,
        hits=hits,
        events=events,
        meta=meta,
    )
    progress(100.0, f"done hits={len(hits)} events={len(events)}")

    logger.info(
        "scan end source=%s scan_input=%s sampled=%s hits=%s events=%s "
        "batch_size=%s batch_runs=%s onnx_batch_dim=%s providers=%s "
        "decode=%.3fs inference=%.3fs frame_save=%.3fs total=%.3fs "
        "sampled_fps=%.2f ms/batch=%.1f ms/frame=%.1f",
        source_path,
        input_path,
        sampled,
        len(hits),
        len(events),
        batch_size,
        batch_runs,
        onnx_batch_dim,
        active_providers,
        t_decode,
        t_inference,
        t_frame_save,
        duration,
        sampled_fps,
        ms_per_batch,
        ms_per_frame,
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
        "meta": meta,
        "status_note": "",
    }
