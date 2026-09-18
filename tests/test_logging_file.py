"""File logging under user log dir (env override for tests)."""

from __future__ import annotations

import logging
from pathlib import Path

from framesentry.core.logging_setup import (
    log_environment_banner,
    reset_logging_for_tests,
    setup_logging,
)


def test_file_handler_creates_and_writes(tmp_path: Path, monkeypatch):
    reset_logging_for_tests()
    monkeypatch.setenv("FRAMESENTRY_LOG_DIR", str(tmp_path))
    log_path = setup_logging()
    assert log_path is not None
    assert log_path.is_file()
    assert log_path.parent == tmp_path

    log_environment_banner(active_providers=["CPUExecutionProvider"])
    logging.getLogger("framesentry.test").info("hello-file-log")
    for h in logging.getLogger().handlers:
        h.flush()

    text = log_path.read_text(encoding="utf-8")
    assert "FrameSentry" in text
    assert "hello-file-log" in text
    assert "CPUExecutionProvider" in text
    reset_logging_for_tests()
