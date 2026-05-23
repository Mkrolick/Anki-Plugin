"""Public API surface for smart_grader.

The single import point for other add-ons that want to calibrate notes via
smart_grader. Keeping the surface narrow lets us evolve the internals
without breaking downstream consumers.
"""
from __future__ import annotations

import importlib.util


def is_available() -> bool:
    """True iff this module can be imported and aqt is on the path.

    book_to_cards calls this before attempting calibrate_note so it can
    silently skip integration when smart_grader isn't installed.
    """
    return importlib.util.find_spec("aqt") is not None


def calibrate_note(note) -> dict:
    """Calibrate a single Anki note in-place.

    Returns a dict {"nid", "threshold", "overlap"}. Raises on hard failures
    (missing fields, API errors); callers should catch and report rather
    than letting one bad note abort a batch.
    """
    from .calibration import _calibrate_one
    return _calibrate_one(note)


def precompute_calibration(*, question: str, reference: str, keywords=None) -> dict:
    """Run the LLM + embedding work for a card without touching Anki.

    Safe to call from a background thread. Returns a dict with the values
    apply_calibration() needs. Useful for batched flows like book_to_cards,
    which calls this off the main thread so insertion can be fast.
    """
    from .calibration import precompute_calibration as _impl
    return _impl(question=question, reference=reference, keywords=keywords)


def apply_calibration(note, calibration: dict) -> dict:
    """Write pre-computed calibration onto an Anki note. Must run on main."""
    from .calibration import apply_calibration as _impl
    return _impl(note, calibration)
