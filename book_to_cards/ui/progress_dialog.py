"""Non-modal progress dialog that subscribes to BookRunner signals."""
from __future__ import annotations

try:
    from aqt.qt import (
        QDialog,
        QLabel,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
        Qt,
    )

    class ProgressDialog(QDialog):
        def __init__(self, runner, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Generating cards…")
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
            self.setModal(False)
            self._runner = runner

            self.stage_label = QLabel("Starting…")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.cost_label = QLabel("$0.00")
            self.stop_button = QPushButton("Stop")
            self.stop_button.clicked.connect(self._on_stop)

            layout = QVBoxLayout(self)
            layout.addWidget(self.stage_label)
            layout.addWidget(self.progress_bar)
            layout.addWidget(self.cost_label)
            layout.addWidget(self.stop_button)
            self.resize(420, 160)

            runner.stage_started.connect(self._on_stage)
            runner.chunk_done.connect(self._on_step)
            runner.topic_done.connect(self._on_step)
            runner.cost_updated.connect(self._on_cost)
            runner.aborted.connect(self._on_aborted)
            runner.finished_with_summary.connect(self._on_done)

        def _on_stage(self, name: str):
            self.stage_label.setText(f"Stage: {name}")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

        def _on_step(self, done: int, total: int):
            if total <= 0:
                return
            pct = int(100 * done / total)
            self.progress_bar.setValue(pct)
            head = self.stage_label.text().split(":")[0]
            self.stage_label.setText(f"{head}: {done}/{total}")

        def _on_cost(self, usd: float):
            self.cost_label.setText(f"${usd:.3f}")

        def _on_stop(self):
            self._runner.request_stop()
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping…")

        def _on_aborted(self, reason: str):
            self.stage_label.setText(f"Aborted: {reason}")
            self.stop_button.setText("Close")
            self.stop_button.setEnabled(True)
            try:
                self.stop_button.clicked.disconnect()
            except TypeError:
                pass
            self.stop_button.clicked.connect(self.accept)

        def _on_done(self, summary: dict):
            inserted = summary.get("inserted", 0)
            errors = summary.get("errors", 0)
            cost = summary.get("cost_usd", 0.0)
            self.stage_label.setText(
                f"Done: {inserted} cards inserted ({errors} errors, ${cost:.3f})"
            )
            self.progress_bar.setValue(100)
            self.stop_button.setText("Close")
            try:
                self.stop_button.clicked.disconnect()
            except TypeError:
                pass
            self.stop_button.clicked.connect(self.accept)
except ImportError:
    ProgressDialog = None  # type: ignore[assignment]
