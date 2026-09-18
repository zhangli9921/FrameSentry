"""DetectionEvent clustering."""

from framesentry.core.events import cluster_hits
from framesentry.core.timecode import format_timestamp
from framesentry.core.types import DetectionHit


def _hit(t: float, cls: str, score: float = 0.5, fi: int = 0) -> DetectionHit:
    return DetectionHit(
        frame_index=fi,
        timestamp_sec=t,
        timestamp_str=format_timestamp(t),
        class_name=cls,
        score=score,
        box=[0, 0, 10, 10],
        frame_filename=None,
    )


def test_cluster_empty():
    assert cluster_hits([]) == []


def test_cluster_merge_within_window():
    hits = [
        _hit(1.0, "FEMALE_BREAST_EXPOSED", 0.4, 10),
        _hit(2.5, "FEMALE_BREAST_EXPOSED", 0.9, 20),
        _hit(3.0, "BUTTOCKS_EXPOSED", 0.5, 30),
    ]
    events = cluster_hits(hits, merge_window_sec=2.0)
    assert len(events) == 1
    assert events[0].count == 3
    assert events[0].max_score == 0.9
    assert events[0].best_frame_index == 20
    assert "FEMALE_BREAST_EXPOSED" in events[0].classes


def test_cluster_split_by_gap():
    hits = [
        _hit(1.0, "ANUS_EXPOSED", 0.5, 1),
        _hit(5.0, "ANUS_EXPOSED", 0.6, 2),
    ]
    events = cluster_hits(hits, merge_window_sec=2.0)
    assert len(events) == 2


def test_raw_hits_preserved_via_indices():
    hits = [_hit(0.0, "MALE_GENITALIA_EXPOSED", 0.7, 0)]
    events = cluster_hits(hits)
    assert events[0].hit_indices == [0]
    assert len(hits) == 1
