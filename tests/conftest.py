"""Pytest fixtures — keep CI offline (no model download, mock ORT session)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def mock_inference_session():
    """Minimal onnxruntime.InferenceSession mock."""
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    inp = MagicMock()
    inp.name = "images"
    session.get_inputs.return_value = [inp]
    session.run.return_value = [MagicMock()]
    return session
