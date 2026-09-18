"""Annotated frame saving with safe filenames."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_frame_filename(
    frame_index: int,
    timestamp_str: str,
    class_name: str,
    score: float,
) -> str:
    """Build ``000023__00-07-42.251__FEMALE_BREAST_EXPOSED__0.55.jpg``."""
    safe_class = _UNSAFE.sub("_", class_name)
    safe_ts = _UNSAFE.sub("-", timestamp_str)
    return f"{frame_index:06d}__{safe_ts}__{safe_class}__{score:.2f}.jpg"


def annotate_and_save_frame(
    image_bgr: np.ndarray,
    detections: list[dict],
    out_path: str | Path,
) -> Path:
    """Draw red boxes + ``CLASS | 0.68`` fully inside image; write JPEG."""
    img = image_bgr.copy()
    h_img, w_img = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for det in detections:
        box = det.get("box") or [0, 0, 0, 0]
        x, y, w, h = [int(v) for v in box[:4]]
        cls = str(det.get("class") or det.get("class_name") or "?")
        score = float(det.get("score", 0.0))
        label = f"{cls} | {score:.2f}"

        x2 = min(w_img - 1, x + max(0, w))
        y2 = min(h_img - 1, y + max(0, h))
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        # Prefer above box; if clipped, place below; if still clipped, clamp inside.
        text_x = x
        text_y = y - 4
        if text_y - th < 0:
            text_y = y2 + th + 4
        if text_y + baseline >= h_img:
            text_y = h_img - baseline - 2
        if text_y - th < 0:
            text_y = th + 2
        if text_x + tw >= w_img:
            text_x = max(0, w_img - tw - 2)
        if text_x < 0:
            text_x = 0

        cv2.rectangle(
            img,
            (text_x, text_y - th - 2),
            (text_x + tw + 2, text_y + baseline),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            img,
            label,
            (text_x + 1, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out
