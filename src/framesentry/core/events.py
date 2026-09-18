"""DetectionEvent clustering from raw hits."""

from __future__ import annotations

from framesentry.core.config import DEFAULT_MERGE_WINDOW_SEC, TARGET_CLASSES
from framesentry.core.timecode import format_timestamp
from framesentry.core.types import DetectionEvent, DetectionHit


def classes_compatible(a: str, b: str) -> bool:
    """Same class, or both in the target filter set (compatible for clustering)."""
    if a == b:
        return True
    return a in TARGET_CLASSES and b in TARGET_CLASSES


def cluster_hits(
    hits: list[DetectionHit],
    merge_window_sec: float = DEFAULT_MERGE_WINDOW_SEC,
) -> list[DetectionEvent]:
    """Cluster adjacent hits with dt <= window and compatible classes.

    Raw hits are never mutated or dropped; this only builds a summary.
    Hits are sorted by timestamp then frame_index before clustering.
    """
    if not hits:
        return []

    ordered = sorted(
        enumerate(hits),
        key=lambda pair: (pair[1].timestamp_sec, pair[1].frame_index, pair[0]),
    )

    events: list[DetectionEvent] = []
    cluster_indices: list[int] = []
    cluster_hits_list: list[DetectionHit] = []

    def flush() -> None:
        nonlocal cluster_indices, cluster_hits_list
        if not cluster_hits_list:
            return
        classes = sorted({h.class_name for h in cluster_hits_list})
        best = max(cluster_hits_list, key=lambda h: h.score)
        start = cluster_hits_list[0]
        end = cluster_hits_list[-1]
        events.append(
            DetectionEvent(
                start_sec=start.timestamp_sec,
                end_sec=end.timestamp_sec,
                start_str=format_timestamp(start.timestamp_sec),
                end_str=format_timestamp(end.timestamp_sec),
                count=len(cluster_hits_list),
                classes=classes,
                max_score=best.score,
                best_frame_index=best.frame_index,
                best_frame_filename=best.frame_filename,
                best_timestamp_sec=best.timestamp_sec,
                hit_indices=list(cluster_indices),
            )
        )
        cluster_indices = []
        cluster_hits_list = []

    for orig_idx, hit in ordered:
        if not cluster_hits_list:
            cluster_indices.append(orig_idx)
            cluster_hits_list.append(hit)
            continue

        last = cluster_hits_list[-1]
        dt = hit.timestamp_sec - last.timestamp_sec
        compatible = any(
            classes_compatible(hit.class_name, existing.class_name)
            for existing in cluster_hits_list
        )
        if dt <= merge_window_sec and compatible:
            cluster_indices.append(orig_idx)
            cluster_hits_list.append(hit)
        else:
            flush()
            cluster_indices.append(orig_idx)
            cluster_hits_list.append(hit)

    flush()
    return events
