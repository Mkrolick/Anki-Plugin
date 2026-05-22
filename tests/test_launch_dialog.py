"""Cost estimator is the only pure-Python part of the launch dialog."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_cost_scales_linearly_with_pages():
    from book_to_cards.ui.launch_dialog import estimate_cost
    low_450, high_450 = estimate_cost(450, skip_calibration=False)
    low_900, high_900 = estimate_cost(900, skip_calibration=False)
    assert abs(low_900 / low_450 - 2.0) < 0.01
    assert abs(high_900 / high_450 - 2.0) < 0.01


def test_skip_calibration_reduces_cost():
    from book_to_cards.ui.launch_dialog import estimate_cost
    full = estimate_cost(450, skip_calibration=False)
    skip = estimate_cost(450, skip_calibration=True)
    assert skip[0] < full[0]
    assert skip[1] < full[1]


def test_central_estimate_matches_spec():
    """450 pages, full calibration → central ≈ $0.53."""
    from book_to_cards.ui.launch_dialog import estimate_cost
    low, high = estimate_cost(450, skip_calibration=False)
    central = (low + high) / 2
    assert abs(central - 0.53) < 0.01
