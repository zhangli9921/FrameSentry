"""results.json read/write — ALWAYS keeps all raw hits."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from framesentry.core.types import DetectionEvent, DetectionHit

logger = logging.getLogger(__name__)


def save_results(
    review_dir: str | Path,
    *,
    video_path: str,
    hits: list[DetectionHit],
    events: list[DetectionEvent],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write results.json with ALL raw hits + events summary. Never drop hits.

    Also regenerates a static ``index.html`` review page alongside frames/.
    """
    review = Path(review_dir)
    review.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "video_path": video_path,
        "hit_count": len(hits),
        "event_count": len(events),
        "hits": [h.to_dict() for h in hits],
        "events": [e.to_dict() for e in events],
        "meta": meta or {},
    }
    out = review / "results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        from framesentry.storage.html_report import write_html_report

        write_html_report(
            review,
            video_path=video_path,
            hits=hits,
            events=events,
            meta=meta,
        )
    except Exception:  # noqa: BLE001 — JSON is primary; HTML must not fail the scan
        logger.exception("Failed to write index.html for review_dir=%s", review)

    return out


def load_results(review_dir: str | Path) -> dict[str, Any]:
    path = Path(review_dir) / "results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    # Rehydrate typed objects for callers that want them
    data["_hits"] = [DetectionHit.from_dict(h) for h in data.get("hits", [])]
    data["_events"] = [DetectionEvent.from_dict(e) for e in data.get("events", [])]
    return data


def results_to_serializable(
    hits: list[DetectionHit],
    events: list[DetectionEvent],
) -> dict[str, Any]:
    """Pure serialize helper for tests."""
    return {
        "hits": [h.to_dict() for h in hits],
        "events": [e.to_dict() for e in events],
    }
