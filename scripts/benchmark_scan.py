#!/usr/bin/env python3
"""Manual benchmark helper for remux + batched scan (not a CI gate).

Example:
  python scripts/benchmark_scan.py --video /path/to/clip.mp4 --device gpu --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="FrameSentry remux+batch scan benchmark")
    parser.add_argument("--video", required=True, help="Source video path")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--intermediate-dir", default="")
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    from framesentry.core.ffmpeg import resolve_ffmpeg
    from framesentry.detectors.nudenet_backend import create_nudenet_backend
    from framesentry.scanner.preprocess import default_intermediate_dir, preprocess_video
    from framesentry.scanner.worker import ScanSettings, scan_video

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise SystemExit("FFmpeg not found (set FRAMESENTRY_FFMPEG or PATH)")

    intermediate_dir = args.intermediate_dir or str(default_intermediate_dir())
    output_root = args.output_root or str(video.parent / "FrameSentryBenchOut")

    t0 = time.perf_counter()
    prep = preprocess_video(video, intermediate_dir, ffmpeg_bin=ffmpeg)
    t_prep = time.perf_counter() - t0

    detector = create_nudenet_backend(device=args.device)
    settings = ScanSettings(
        sample_fps=args.sample_fps,
        device=args.device,
        output_root=output_root,
        save_frames=False,
        batch_size=args.batch_size,
        intermediate_dir=intermediate_dir,
    )
    t1 = time.perf_counter()
    result = scan_video(
        str(video),
        detector,
        settings,
        scan_input_path=prep.intermediate_path,
        preprocess_meta={
            "source_video_path": str(video),
            "scan_input_path": prep.intermediate_path,
            "intermediate_path": prep.intermediate_path,
            "preprocess_duration_sec": prep.duration_sec,
            "ffmpeg_exit_code": prep.exit_code,
        },
    )
    t_scan = time.perf_counter() - t1

    summary = {
        "video": str(video),
        "device": args.device,
        "batch_size": args.batch_size,
        "preprocess_sec": round(t_prep, 3),
        "scan_sec": round(t_scan, 3),
        "total_sec": round(t_prep + t_scan, 3),
        "meta": result.get("meta"),
        "hit_count": result.get("hit_count"),
        "sampled_frames": result.get("sampled_frames"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
