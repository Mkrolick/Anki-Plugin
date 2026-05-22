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
