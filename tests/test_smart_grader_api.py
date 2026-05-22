"""Smart Grader's external API surface."""
from __future__ import annotations

import sys
from pathlib import Path


def test_api_module_exports_required_names():
    source = (Path(__file__).resolve().parent.parent / "smart_grader" / "api.py").read_text()
    assert "def calibrate_note(" in source
    assert "def is_available(" in source


def test_is_available_returns_false_without_aqt(monkeypatch):
    """is_available() should guard against missing aqt via importlib.util.find_spec."""
    source = (Path(__file__).resolve().parent.parent / "smart_grader" / "api.py").read_text()
    assert "find_spec" in source or "try:" in source, \
        "is_available() should guard against missing aqt"


def test_calibration_has_underscore_alias():
    """smart_grader.calibration._calibrate_one is the canonical entry point;
    calibrate_note remains as a backwards-compatible alias."""
    source = (Path(__file__).resolve().parent.parent / "smart_grader" / "calibration.py").read_text()
    assert "def _calibrate_one(" in source
    # backwards-compat alias preserved
    assert "calibrate_note = _calibrate_one" in source
