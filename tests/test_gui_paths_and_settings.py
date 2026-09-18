"""GUI path defaults, settings, CPU batch=1, worker finished race."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog

from framesentry.core.config import BATCH_SIZE_CHOICES, DEFAULT_BATCH_SIZE
from framesentry.scanner.preprocess import default_intermediate_dir
from framesentry.ui.main_window import MainWindow
from framesentry.ui.scan_controller import ScanWorker


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_mainwindow_intermediate_dir_default_without_choose(qapp):
    win = MainWindow()
    try:
        assert hasattr(win, "_intermediate_dir")
        assert win._intermediate_dir == str(default_intermediate_dir())
        settings = win._current_settings()
        assert settings.intermediate_dir == str(default_intermediate_dir())
    finally:
        win.close()


def test_choose_intermediate_dialog_initial_dir(qapp, monkeypatch):
    win = MainWindow()
    seen = {}

    def fake_get(parent, title, initial):
        seen["initial"] = initial
        return "/tmp/custom_inter"

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(fake_get))
    try:
        win._on_choose_intermediate()
        assert seen["initial"] == str(default_intermediate_dir())
        assert win._intermediate_dir == "/tmp/custom_inter"
        assert win._current_settings().intermediate_dir == "/tmp/custom_inter"
    finally:
        win.close()


def test_cpu_batch_forced_to_1(qapp):
    win = MainWindow()
    try:
        win.radio_cpu.setChecked(True)
        # Select a GPU-sized batch in the combo; CPU must still force 1
        idx = list(BATCH_SIZE_CHOICES).index(16)
        win.batch_combo.setCurrentIndex(idx)
        settings = win._current_settings()
        assert settings.device == "cpu"
        assert settings.batch_size == 1
    finally:
        win.close()


def test_gpu_batch_choices_flow_into_settings(qapp):
    win = MainWindow()
    try:
        win.radio_gpu.setChecked(True)
        win.radio_gpu.setEnabled(True)
        for b in BATCH_SIZE_CHOICES:
            idx = list(BATCH_SIZE_CHOICES).index(b)
            win.batch_combo.setCurrentIndex(idx)
            settings = win._current_settings()
            assert settings.batch_size == b
        assert DEFAULT_BATCH_SIZE == 16
    finally:
        win.close()


def test_worker_finished_only_clears_matching_instance(qapp):
    win = MainWindow()
    try:
        w1 = ScanWorker(win)
        w2 = ScanWorker(win)
        win._worker = w2
        # Simulate w1 finishing after w2 replaced it
        win._on_worker_thread_finished(w1)
        assert win._worker is w2
        win._on_worker_thread_finished(w2)
        assert win._worker is None
    finally:
        win.close()
