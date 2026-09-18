"""Safe filenames and JSON serialize."""

import json
from pathlib import Path

from framesentry.core.timecode import format_timestamp
from framesentry.core.types import DetectionHit
from framesentry.core.events import cluster_hits
from framesentry.storage.frames import safe_frame_filename
from framesentry.storage.results import results_to_serializable, save_results, load_results


def test_safe_filename():
    name = safe_frame_filename(23, "00-07-42.251", "FEMALE_BREAST_EXPOSED", 0.55)
    assert name == "000023__00-07-42.251__FEMALE_BREAST_EXPOSED__0.55.jpg"
    assert "/" not in name
    assert "\\" not in name


def test_safe_filename_strips_unsafe():
    name = safe_frame_filename(1, "00-00-01.000", "A/B:C", 0.1)
    for ch in '<>:"|?*':
        assert ch not in name
    assert "/" not in name


def test_json_serialize_roundtrip(tmp_path: Path):
    hits = [
        DetectionHit(
            frame_index=1,
            timestamp_sec=1.5,
            timestamp_str=format_timestamp(1.5),
            class_name="BUTTOCKS_EXPOSED",
            score=0.66,
            box=[1, 2, 3, 4],
            frame_filename="x.jpg",
        )
    ]
    events = cluster_hits(hits)
    payload = results_to_serializable(hits, events)
    raw = json.dumps(payload)
    loaded = json.loads(raw)
    assert len(loaded["hits"]) == 1
    assert loaded["hits"][0]["class_name"] == "BUTTOCKS_EXPOSED"

    save_results(tmp_path, video_path="/v.mp4", hits=hits, events=events)
    data = load_results(tmp_path)
    assert len(data["_hits"]) == 1
    assert len(data["hits"]) == 1  # NEVER drop raw hits
