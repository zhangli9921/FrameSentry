"""GUI modules importable / lightweight init (no real mouse DnD)."""

from __future__ import annotations

import os

import pytest

# Headless Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_gui_modules_importable():
    from framesentry.ui import main_window, review_panel, scan_controller

    assert main_window.MainWindow is not None
    assert review_panel.ReviewPanel is not None
    assert scan_controller.ScanWorker is not None


def test_main_window_init_and_dnd_flags():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from framesentry.ui.main_window import DropZoneLabel, MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert win.acceptDrops() is True
    assert hasattr(win, "dragEnterEvent")
    assert hasattr(win, "dropEvent")
    assert hasattr(win, "dragLeaveEvent")
    zone = DropZoneLabel()
    zone.set_active()
    assert "释放" in zone.text()
    zone.set_idle()
    win.close()


def test_review_panel_init():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from framesentry.ui.review_panel import ReviewPanel

    app = QApplication.instance() or QApplication([])
    panel = ReviewPanel()
    panel.clear()
    assert panel.event_list.count() == 0
