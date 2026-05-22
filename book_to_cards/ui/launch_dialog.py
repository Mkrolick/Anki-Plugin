"""Launch dialog: pick deck + cost estimate + max-cost ceiling."""
from __future__ import annotations

import os
from dataclasses import dataclass

# Cost coefficients derived from spec §7 (450-page reference run).
_BASE_PAGES = 450
_BASE_COST_USD = 0.53
_CALIBRATION_SHARE = 0.30 / 0.53


@dataclass
class LaunchChoice:
    deck_name: str
    max_cost_usd: float
    skip_calibration: bool


def estimate_cost(page_count: int, skip_calibration: bool) -> tuple[float, float]:
    """Return (low, high) USD estimate. ±50% band on the central estimate."""
    central = _BASE_COST_USD * (page_count / _BASE_PAGES)
    if skip_calibration:
        central *= 1 - _CALIBRATION_SHARE
    return central * 0.5, central * 1.5


# The Qt widget is only constructable inside Anki. Importing aqt here at
# module top is fine because tests of estimate_cost can import this file
# even without aqt — the import below is guarded.
try:
    from aqt import mw  # noqa: F401
    from aqt.qt import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QLabel,
        QVBoxLayout,
    )

    class LaunchDialog(QDialog):
        def __init__(self, pdf_path: str, page_count: int, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Generate cards from PDF")
            self._pdf_path = pdf_path
            self._page_count = page_count

            form = QFormLayout()

            default_deck = self._default_deck_name(pdf_path)
            self.deck_combo = QComboBox()
            self.deck_combo.setEditable(True)
            try:
                from aqt import mw as _mw
                existing = [d.name for d in _mw.col.decks.all_names_and_ids()]
            except Exception:
                existing = []
            existing = sorted(set(existing) | {default_deck})
            self.deck_combo.addItems(existing)
            self.deck_combo.setCurrentText(default_deck)
            form.addRow("Target deck:", self.deck_combo)

            self.skip_calibration = QCheckBox("Skip per-card calibration (cheaper, lower quality)")
            form.addRow("", self.skip_calibration)

            self.cost_label = QLabel("")
            form.addRow("Estimated cost:", self.cost_label)

            self.ceiling = QDoubleSpinBox()
            self.ceiling.setRange(0.1, 1000.0)
            self.ceiling.setDecimals(2)
            self.ceiling.setValue(5.0)
            self.ceiling.setPrefix("$")
            form.addRow("Max-cost ceiling:", self.ceiling)

            self.skip_calibration.toggled.connect(self._refresh_cost)
            self._refresh_cost()

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)

            layout = QVBoxLayout(self)
            layout.addLayout(form)
            layout.addWidget(buttons)

        def _default_deck_name(self, pdf_path: str) -> str:
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            return f"Book - {base}"

        def _refresh_cost(self):
            low, high = estimate_cost(self._page_count, self.skip_calibration.isChecked())
            self.cost_label.setText(f"${low:.2f} – ${high:.2f}")

        def choice(self) -> LaunchChoice:
            return LaunchChoice(
                deck_name=self.deck_combo.currentText().strip() or self._default_deck_name(self._pdf_path),
                max_cost_usd=float(self.ceiling.value()),
                skip_calibration=self.skip_calibration.isChecked(),
            )
except ImportError:
    # aqt missing — only estimate_cost is available for headless testing.
    LaunchDialog = None  # type: ignore[assignment]
