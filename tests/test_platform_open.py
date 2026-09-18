"""Platform open helpers — mock subprocess / desktop services (no real Explorer)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from framesentry.core.platform_open import (
    PlatformOpenError,
    logs_directory,
    open_directory,
    open_html_in_browser,
    open_logs_directory,
    open_path_with_default_app,
    reveal_in_file_manager,
)


def test_reveal_windows_uses_explorer_select(tmp_path: Path, monkeypatch):
    f = tmp_path / "图 片.jpg"
    f.write_bytes(b"x")
    monkeypatch.setattr(sys, "platform", "win32")
    with patch("framesentry.core.platform_open.subprocess.run") as run:
        reveal_in_file_manager(f)
        run.assert_called_once()
        args = run.call_args[0][0]
        assert args[0] == "explorer"
        assert args[1].startswith("/select,")
        assert "图 片.jpg" in args[1] or str(f.resolve()) in args[1] or "图" in args[1]


def test_reveal_non_windows_opens_parent(tmp_path: Path, monkeypatch):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    monkeypatch.setattr(sys, "platform", "linux")
    with patch("framesentry.core.platform_open.open_path_with_default_app") as opener:
        reveal_in_file_manager(f)
        opener.assert_called_once()
        assert Path(opener.call_args[0][0]) == f.parent.resolve() or Path(
            opener.call_args[0][0]
        ).resolve() == f.parent.resolve()


def test_open_missing_raises(tmp_path: Path):
    with pytest.raises(PlatformOpenError):
        open_path_with_default_app(tmp_path / "nope.jpg")


def test_open_html_uses_qdesktopservices(tmp_path: Path):
    index = tmp_path / "index.html"
    index.write_text("<html></html>", encoding="utf-8")
    mock_qds = MagicMock()
    mock_qds.openUrl.return_value = True
    with patch.dict(
        "sys.modules",
        {
            # Ensure imports inside function see mocks via real PySide6 if present
        },
    ):
        with patch("PySide6.QtGui.QDesktopServices.openUrl", return_value=True) as open_url:
            with patch("PySide6.QtCore.QUrl.fromLocalFile", return_value=MagicMock()) as from_local:
                open_html_in_browser(index)
                from_local.assert_called()
                open_url.assert_called()


def test_logs_directory_matches_logging_setup(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FRAMESENTRY_LOG_DIR", str(tmp_path / "logs"))
    assert logs_directory() == tmp_path / "logs"


def test_open_logs_directory_mocked(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FRAMESENTRY_LOG_DIR", str(tmp_path / "logs"))
    with patch("framesentry.core.platform_open.open_directory") as od:
        opened = open_logs_directory()
        assert opened == tmp_path / "logs"
        od.assert_called_once()


def test_open_directory_file_opens_parent(tmp_path: Path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"1")
    with patch("framesentry.core.platform_open.open_path_with_default_app") as opener:
        open_directory(f)
        called = Path(opener.call_args[0][0])
        assert called.resolve() == tmp_path.resolve()
