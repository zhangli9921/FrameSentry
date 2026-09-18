"""Static HTML review page generation."""

from __future__ import annotations

from pathlib import Path

from framesentry.core.types import DetectionEvent, DetectionHit
from framesentry.storage.html_report import (
    build_review_html,
    group_hits_by_frame,
    write_html_report,
)
from framesentry.storage.results import load_results, save_results


def _hit(idx: int, cls: str, score: float, fname: str) -> DetectionHit:
    return DetectionHit(
        frame_index=idx,
        timestamp_sec=float(idx),
        timestamp_str=f"00-00-{idx:02d}.000",
        class_name=cls,
        score=score,
        box=[1, 2, 3, 4],
        frame_filename=fname,
    )


def test_group_hits_same_frame_one_tile():
    fname = "000010__00-00-10.000__X__0.90.jpg"
    hits = [
        _hit(10, "FEMALE_BREAST_EXPOSED", 0.9, fname),
        _hit(10, "BUTTOCKS_EXPOSED", 0.8, fname),
        _hit(20, "ANUS_EXPOSED", 0.7, "other.jpg"),
    ]
    groups = group_hits_by_frame(hits)
    assert len(groups) == 2
    assert len(groups[0]["detections"]) == 2
    assert groups[0]["frame_index"] == 10


def test_write_html_relative_paths_and_header(tmp_path: Path):
    fname = "000001__00-00-01.000__FEMALE_BREAST_EXPOSED__0.90.jpg"
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / fname).write_bytes(b"fake")
    hits = [_hit(1, "FEMALE_BREAST_EXPOSED", 0.9, fname)]
    events = [
        DetectionEvent(
            start_sec=1.0,
            end_sec=1.0,
            start_str="00-00-01.000",
            end_str="00-00-01.000",
            count=1,
            classes=["FEMALE_BREAST_EXPOSED"],
            max_score=0.9,
            best_frame_index=1,
            best_frame_filename=fname,
            best_timestamp_sec=1.0,
            hit_indices=[0],
        )
    ]
    meta = {
        "sample_fps": 2.0,
        "threshold": 0.35,
        "sampled_frames": 100,
        "duration_sec": 1.23,
        "warnings": ["w1"],
    }
    out = write_html_report(
        tmp_path,
        video_path=r"D:\videos\测试\clip.mp4",
        hits=hits,
        events=events,
        meta=meta,
    )
    assert out.name == "index.html"
    html = out.read_text(encoding="utf-8")
    assert "frames/" + fname in html
    assert "data:image" not in html
    assert "base64," not in html.lower()
    assert "clip.mp4" in html or "测试" in html
    assert "2.0" in html
    assert "0.35" in html
    assert "100" in html
    assert "1.23" in html
    assert "href=\"frames/" in html


def test_html_empty_state_bilingual(tmp_path: Path):
    html = build_review_html(
        video_path="/v.mp4",
        hits=[],
        events=[],
        meta={"sample_fps": 2, "threshold": 0.35, "sampled_frames": 10, "duration_sec": 0.1},
    )
    assert "未检测到目标违规帧" in html
    assert "No target violation" in html


def test_save_results_also_writes_index(tmp_path: Path):
    hits = [_hit(1, "BUTTOCKS_EXPOSED", 0.5, "a.jpg")]
    events: list[DetectionEvent] = []
    save_results(tmp_path, video_path="/v.mp4", hits=hits, events=events, meta={"sample_fps": 2})
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "index.html").is_file()
    data = load_results(tmp_path)
    assert len(data["hits"]) == 1
