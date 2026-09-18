"""Static HTML review page generator (no server, relative paths only)."""

from __future__ import annotations

import html
from collections import OrderedDict
from pathlib import Path
from typing import Any

from framesentry.core.types import DetectionEvent, DetectionHit


def group_hits_by_frame(hits: list[DetectionHit]) -> list[dict[str, Any]]:
    """Collapse multi-detections on the same frame into one tile.

    Order follows first appearance of each frame_index / frame_filename.
    """
    groups: OrderedDict[tuple[int, str | None], dict[str, Any]] = OrderedDict()
    for hit in hits:
        key = (hit.frame_index, hit.frame_filename)
        if key not in groups:
            groups[key] = {
                "frame_index": hit.frame_index,
                "timestamp_sec": hit.timestamp_sec,
                "timestamp_str": hit.timestamp_str,
                "frame_filename": hit.frame_filename,
                "detections": [],
            }
        groups[key]["detections"].append(
            {"class_name": hit.class_name, "score": hit.score, "box": list(hit.box)}
        )
    return list(groups.values())


def _fmt_score(score: float) -> str:
    return f"{float(score):.2f}"


def _card_html(group: dict[str, Any]) -> str:
    fname = group.get("frame_filename") or ""
    rel = f"frames/{html.escape(fname)}" if fname else ""
    ts = html.escape(str(group.get("timestamp_str") or ""))
    idx = int(group.get("frame_index") or 0)
    dets = group.get("detections") or []
    labels = []
    for d in dets:
        labels.append(
            f"{html.escape(str(d.get('class_name') or '?'))} "
            f"({_fmt_score(float(d.get('score') or 0.0))})"
        )
    labels_html = "<br/>".join(labels) if labels else "(no labels)"
    if fname:
        img = (
            f'<a href="{rel}" target="_blank" rel="noopener">'
            f'<img src="{rel}" alt="frame {idx}" loading="lazy"/></a>'
        )
    else:
        img = '<div class="missing">无帧图 / missing frame</div>'
    return (
        f'<article class="card" data-frame-index="{idx}">'
        f"{img}"
        f'<div class="meta">'
        f"<div class=\"ts\">{ts}</div>"
        f"<div class=\"labels\">{labels_html}</div>"
        f"<div class=\"idx\">frame #{idx}</div>"
        f"</div></article>"
    )


def _event_section_html(
    events: list[DetectionEvent],
    hits: list[DetectionHit],
) -> str:
    if not events:
        return "<p class=\"empty\">无事件 / No events</p>"
    parts: list[str] = []
    for i, ev in enumerate(events):
        classes = ", ".join(html.escape(c) for c in (ev.classes or [])) or "?"
        header = (
            f"<h3>事件 {i + 1}: {html.escape(ev.start_str)} → "
            f"{html.escape(ev.end_str)} | {classes} | "
            f"max={_fmt_score(ev.max_score)} | n={ev.count}</h3>"
        )
        # Prefer best frame; also list unique frames in the event.
        seen: set[tuple[int, str | None]] = set()
        cards: list[str] = []
        # Best frame first
        best_hits = [
            h
            for h in hits
            if h.frame_index == ev.best_frame_index
            and (h.frame_filename == ev.best_frame_filename or ev.best_frame_filename is None)
        ]
        if not best_hits and ev.best_frame_filename:
            # Synthetic group from event fields
            group = {
                "frame_index": ev.best_frame_index,
                "timestamp_sec": ev.best_timestamp_sec,
                "timestamp_str": ev.start_str,
                "frame_filename": ev.best_frame_filename,
                "detections": [
                    {"class_name": c, "score": ev.max_score, "box": [0, 0, 0, 0]}
                    for c in (ev.classes or ["?"])
                ],
            }
            cards.append(_card_html(group))
            seen.add((ev.best_frame_index, ev.best_frame_filename))
        event_hit_list = [
            hits[j] for j in ev.hit_indices if 0 <= j < len(hits)
        ]
        # Group event hits by frame
        for group in group_hits_by_frame(event_hit_list):
            key = (group["frame_index"], group["frame_filename"])
            if key in seen:
                continue
            seen.add(key)
            cards.append(_card_html(group))
        # If best was among event hits, ensure it appears — already covered by loop;
        # if best_hits existed, include via grouping above.
        if best_hits:
            for group in group_hits_by_frame(best_hits):
                key = (group["frame_index"], group["frame_filename"])
                if key not in seen:
                    cards.insert(0, _card_html(group))
                    seen.add(key)
        parts.append(
            f'<section class="event" id="event-{i}">'
            f"{header}<div class=\"grid\">{''.join(cards) or '<p class=\"empty\">无帧</p>'}"
            f"</div></section>"
        )
    return "\n".join(parts)


def build_review_html(
    *,
    video_path: str,
    hits: list[DetectionHit],
    events: list[DetectionEvent],
    meta: dict[str, Any] | None = None,
) -> str:
    """Return a complete static HTML document (UTF-8)."""
    meta = meta or {}
    groups = group_hits_by_frame(hits)
    sample_fps = meta.get("sample_fps", "")
    threshold = meta.get("threshold", "")
    sampled = meta.get("sampled_frames", "")
    duration = meta.get("duration_sec", "")
    warnings = meta.get("warnings") or []
    warning_count = len(warnings) if isinstance(warnings, list) else int(warnings or 0)

    header_rows = [
        ("源视频 / Source", str(video_path)),
        ("采样 FPS / Sample FPS", str(sample_fps)),
        ("阈值 / Threshold", str(threshold)),
        ("采样帧数 / Sampled frames", str(sampled)),
        ("原始命中 / Raw hits", str(len(hits))),
        ("事件数 / Events", str(len(events))),
        ("扫描耗时 / Duration (s)", str(duration)),
        ("警告数 / Warnings", str(warning_count)),
    ]
    header_html = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
        for k, v in header_rows
    )

    if groups:
        all_cards = "".join(_card_html(g) for g in groups)
        all_section = f'<div class="grid" id="all-hits-grid">{all_cards}</div>'
    else:
        all_section = (
            '<div class="empty-state" id="all-hits-grid">'
            "<p>未检测到目标违规帧 / No target violation frames detected</p>"
            "</div>"
        )

    events_section = _event_section_html(events, hits)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FrameSentry Review — {html.escape(Path(video_path).name if video_path else 'review')}</title>
<style>
:root {{
  --bg: #0f1115;
  --card: #1a1d24;
  --text: #e8eaed;
  --muted: #9aa0a6;
  --accent: #4fc3f7;
  --border: #2a2f3a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 16px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--bg); color: var(--text);
}}
h1 {{ font-size: 1.25rem; margin: 0 0 12px; }}
h2 {{ font-size: 1.05rem; margin: 24px 0 8px; color: var(--accent); }}
h3 {{ font-size: 0.95rem; margin: 16px 0 8px; color: var(--muted); }}
.header table {{
  border-collapse: collapse; width: 100%; max-width: 960px;
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
}}
.header th, .header td {{
  text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border);
  vertical-align: top; word-break: break-all;
}}
.header th {{ width: 220px; color: var(--muted); font-weight: 600; }}
.tabs {{ margin: 16px 0 8px; display: flex; gap: 8px; flex-wrap: wrap; }}
.tabs a {{
  color: var(--text); text-decoration: none; padding: 6px 12px;
  border: 1px solid var(--border); border-radius: 16px; background: var(--card);
}}
.tabs a:hover {{ border-color: var(--accent); color: var(--accent); }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}}
.card {{
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  overflow: hidden; display: flex; flex-direction: column;
}}
.card img {{
  width: 100%; height: auto; display: block; background: #000;
  aspect-ratio: 16 / 9; object-fit: contain;
}}
.card .meta {{ padding: 8px 10px; font-size: 0.85rem; }}
.card .ts {{ color: var(--accent); font-weight: 600; }}
.card .labels {{ margin-top: 4px; color: var(--text); }}
.card .idx {{ margin-top: 4px; color: var(--muted); font-size: 0.75rem; }}
.missing {{
  padding: 40px 8px; text-align: center; color: var(--muted); background: #111;
}}
.empty-state, .empty {{
  padding: 32px; text-align: center; color: var(--muted);
  border: 1px dashed var(--border); border-radius: 8px; background: var(--card);
}}
.panel {{ display: none; }}
.panel:target, #panel-all:not(:has(~ .panel:target)) {{ display: block; }}
/* Fallback when :has unsupported: show all by default via JS-less first panel */
#panel-all {{ display: block; }}
#panel-events:target {{ display: block; }}
#panel-events:target ~ #panel-all,
body:has(#panel-events:target) #panel-all {{ display: none; }}
</style>
</head>
<body>
<header class="header">
  <h1>FrameSentry 审核页 / Review</h1>
  <table>{header_html}</table>
</header>
<nav class="tabs">
  <a href="#panel-all">全部命中 / All hits</a>
  <a href="#panel-events">按事件 / By event</a>
</nav>
<section id="panel-all" class="panel">
  <h2>全部命中帧 / All hit frames</h2>
  {all_section}
</section>
<section id="panel-events" class="panel">
  <h2>按事件分组 / By event</h2>
  {events_section}
</section>
<script>
(function() {{
  function show(id) {{
    var all = document.getElementById('panel-all');
    var ev = document.getElementById('panel-events');
    if (!all || !ev) return;
    if (id === 'panel-events') {{
      all.style.display = 'none';
      ev.style.display = 'block';
    }} else {{
      all.style.display = 'block';
      ev.style.display = 'none';
    }}
  }}
  function applyHash() {{
    var h = (location.hash || '#panel-all').replace(/^#/, '');
    show(h === 'panel-events' ? 'panel-events' : 'panel-all');
  }}
  window.addEventListener('hashchange', applyHash);
  applyHash();
}})();
</script>
</body>
</html>
"""


def write_html_report(
    review_dir: str | Path,
    *,
    video_path: str,
    hits: list[DetectionHit],
    events: list[DetectionEvent],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``index.html`` under ``review_dir``. Returns the path written."""
    review = Path(review_dir)
    review.mkdir(parents=True, exist_ok=True)
    out = review / "index.html"
    out.write_text(
        build_review_html(
            video_path=video_path,
            hits=hits,
            events=events,
            meta=meta,
        ),
        encoding="utf-8",
    )
    return out
