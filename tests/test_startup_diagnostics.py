"""Startup ordering, watchdog, excepthook, ORT probe signals."""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / "src" / "framesentry" / "main.py"


def test_main_py_does_not_import_heavy_at_module_level():
    """Parse main.py: module-level imports must not include heavy deps / MainWindow."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    forbidden_tops = {"PySide6", "cv2", "numpy", "onnxruntime", "nudenet"}
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in forbidden_tops:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0] if mod else ""
            if top in forbidden_tops:
                found.add(mod)
            if mod.startswith("framesentry.ui.main_window") or mod == "framesentry.ui.main_window":
                found.add(mod)
    assert not found, f"heavy imports at module level: {found}"


def test_main_source_orders_logging_before_pyside():
    src = MAIN_PY.read_text(encoding="utf-8")
    assert "setup_logging()" in src
    assert src.index("setup_logging()") < src.index("importing PySide6")
    assert "main window shown" in src
    assert "watchdog" in src.lower()


def test_watchdog_cancelled_when_window_shown(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMESENTRY_LOG_DIR", str(tmp_path))
    from framesentry.core.logging_setup import reset_logging_for_tests, setup_logging
    from framesentry import main as startup

    reset_logging_for_tests()
    setup_logging()
    startup.reset_startup_flags_for_tests()

    dumped = {"n": 0}

    def fake_dump(reason: str = "") -> None:
        dumped["n"] += 1

    monkeypatch.setattr(startup, "_dump_all_thread_stacks", fake_dump)
    timer = startup.start_startup_watchdog(timeout_sec=0.2)
    startup.mark_main_window_shown()
    startup.cancel_startup_watchdog(timer)
    time.sleep(0.35)
    assert dumped["n"] == 0


def test_watchdog_dumps_when_not_shown(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMESENTRY_LOG_DIR", str(tmp_path))
    from framesentry.core.logging_setup import reset_logging_for_tests, setup_logging
    from framesentry import main as startup

    reset_logging_for_tests()
    setup_logging()
    startup.reset_startup_flags_for_tests()

    dumped = {"n": 0, "reason": ""}

    def fake_dump(reason: str = "") -> None:
        dumped["n"] += 1
        dumped["reason"] = reason

    monkeypatch.setattr(startup, "_dump_all_thread_stacks", fake_dump)
    timer = startup.start_startup_watchdog(timeout_sec=0.15)
    time.sleep(0.35)
    timer.cancel()
    assert dumped["n"] == 1
    assert "watchdog" in dumped["reason"].lower()


def test_dump_all_thread_stacks_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMESENTRY_LOG_DIR", str(tmp_path))
    from framesentry.core.logging_setup import reset_logging_for_tests, setup_logging
    from framesentry import main as startup

    reset_logging_for_tests()
    log_path = setup_logging()
    startup.dump_all_thread_stacks("unit-test dump")
    assert log_path is not None
    text = Path(log_path).read_text(encoding="utf-8")
    assert "unit-test dump" in text
    assert "thread id=" in text


def test_excepthook_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMESENTRY_LOG_DIR", str(tmp_path))
    from framesentry.core.logging_setup import reset_logging_for_tests, setup_logging
    from framesentry import main as startup

    reset_logging_for_tests()
    setup_logging()
    startup._install_excepthooks()
    assert sys.excepthook is not sys.__excepthook__
    try:
        raise RuntimeError("hook-test")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())


def test_ort_probe_worker_signals():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    from framesentry.detectors.base import OrtProviderInfo
    from framesentry.ui.ort_probe import OrtProbeWorker

    worker = OrtProbeWorker()
    results: list = []
    errors: list = []
    worker.finished.connect(results.append)
    worker.failed.connect(errors.append)

    fake = OrtProviderInfo(
        available_providers=("CPUExecutionProvider",),
        cuda_listed=False,
        gpu_mode_usable=False,
    )
    with patch(
        "framesentry.detectors.nudenet_backend.get_ort_provider_info",
        return_value=fake,
    ):
        worker.run()

    assert results
    assert not errors
    assert results[0].cuda_listed is False
