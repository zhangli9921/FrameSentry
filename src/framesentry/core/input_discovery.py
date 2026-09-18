"""Discover video paths from files/folders (shared by buttons and DnD).

GUI must call this module only — no duplicate discovery rules in the UI layer.
Never modifies or copies source files.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from framesentry.core.config import VIDEO_EXTENSIONS


def normalize_video_key(path: str | Path) -> str:
    """Absolute, normpath key; case-insensitive on Windows (normcase/casefold)."""
    p = Path(path).expanduser()
    try:
        resolved = p.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = p.absolute()
    key = os.path.normpath(str(resolved))
    if sys.platform == "win32":
        key = os.path.normcase(key)
        key = key.casefold()
    return key


def is_supported_video(path: str | Path) -> bool:
    """True if path looks like a supported video by extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


@dataclass
class DiscoveryResult:
    """Outcome of expanding user-provided paths into unique video files."""

    videos: list[str] = field(default_factory=list)
    added: int = 0
    ignored: int = 0
    duplicates_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def toast_message(self) -> str:
        """Short Chinese toast for GUI feedback."""
        parts = [f"已添加 {self.added} 个视频"]
        if self.ignored:
            parts.append(f"忽略 {self.ignored} 个不支持项目")
        if self.duplicates_skipped:
            parts.append(f"跳过 {self.duplicates_skipped} 个重复")
        return "，".join(parts)


def discover_videos(
    paths: list[str] | list[Path],
    *,
    existing: set[str] | None = None,
) -> DiscoveryResult:
    """Expand files/folders into unique supported video absolute paths.

    - Files: accept if supported extension.
    - Folders: recursive walk; only supported videos.
    - Unsupported / missing / permission errors: skip safely (ignored++).
    - Dedup against ``existing`` keys and within the batch (normalize_video_key).
    - Never aborts the whole batch on one bad item.
    """
    result = DiscoveryResult()
    seen: set[str] = set(existing or ())

    for raw in paths:
        try:
            p = Path(raw).expanduser()
        except (TypeError, ValueError, OSError) as exc:
            result.ignored += 1
            result.errors.append(f"invalid path {raw!r}: {exc}")
            continue

        try:
            if not p.exists():
                result.ignored += 1
                result.errors.append(f"missing: {p}")
                continue
        except OSError as exc:
            result.ignored += 1
            result.errors.append(f"cannot access {p}: {exc}")
            continue

        try:
            if p.is_file():
                _consider_file(p, seen, result)
            elif p.is_dir():
                _walk_dir(p, seen, result)
            else:
                result.ignored += 1
                result.errors.append(f"unsupported node type: {p}")
        except OSError as exc:
            result.ignored += 1
            result.errors.append(f"error on {p}: {exc}")

    return result


def _consider_file(p: Path, seen: set[str], result: DiscoveryResult) -> None:
    if not is_supported_video(p):
        result.ignored += 1
        return
    try:
        abs_path = str(p.resolve(strict=False))
    except (OSError, RuntimeError):
        abs_path = str(p.absolute())
    key = normalize_video_key(abs_path)
    if key in seen:
        result.duplicates_skipped += 1
        return
    seen.add(key)
    result.videos.append(abs_path)
    result.added += 1


def _walk_dir(root: Path, seen: set[str], result: DiscoveryResult) -> None:
    try:
        walker = os.walk(root, followlinks=False)
    except OSError as exc:
        result.ignored += 1
        result.errors.append(f"cannot walk {root}: {exc}")
        return

    for dirpath, _dirnames, filenames in walker:
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                if not fp.is_file():
                    continue
            except OSError:
                result.ignored += 1
                continue
            if not is_supported_video(fp):
                # non-videos inside folders: ignore quietly (count as ignored)
                result.ignored += 1
                continue
            _consider_file(fp, seen, result)
