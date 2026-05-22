"""Book to Cards for Anki — PDF ingest → auto-generated flashcards.

Loaded by Anki at startup. Registers the Tools menu entry when aqt is
available; importable without aqt so the pipeline modules can be unit-
tested headlessly.
"""
from __future__ import annotations

import os
import sys

_VENDOR = os.path.join(os.path.dirname(__file__), "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

try:
    from aqt import mw
    from aqt.qt import QAction, QFileDialog
    from aqt.utils import showInfo
    _AQT_AVAILABLE = True
except ImportError:
    _AQT_AVAILABLE = False


if _AQT_AVAILABLE:
    # Module-level holders so the QThread + dialog don't get garbage-
    # collected the moment _on_generate returns.
    _active_runner = None
    _active_dialog = None

    def _on_generate():
        global _active_runner, _active_dialog
        from .config import get_config
        from .pdf_text import extract_pages
        from .runner import BookRunner
        from .ui.launch_dialog import LaunchDialog
        from .ui.progress_dialog import ProgressDialog

        if not get_config().get("openai_api_key"):
            showInfo(
                "Set your OpenAI API key in Tools → Add-ons → Book to Cards → Config first."
            )
            return

        pdf_path, _ = QFileDialog.getOpenFileName(
            mw, "Choose a PDF to ingest", "", "PDF files (*.pdf)"
        )
        if not pdf_path:
            return

        try:
            pages = extract_pages(pdf_path)
        except Exception as e:
            showInfo(f"Couldn't read that PDF: {e}")
            return
        if not pages:
            showInfo("No useful text found in that PDF.")
            return

        dlg = LaunchDialog(pdf_path, len(pages))
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        choice = dlg.choice()

        _active_runner = BookRunner(
            pdf_path=pdf_path,
            deck_name=choice.deck_name,
            max_cost_usd=choice.max_cost_usd,
            skip_calibration=choice.skip_calibration,
        )
        _active_dialog = ProgressDialog(_active_runner, parent=mw)
        _active_dialog.show()
        _active_runner.start()

    def _install_menu():
        action = QAction("Generate cards from PDF…", mw)
        action.triggered.connect(_on_generate)
        mw.form.menuTools.addAction(action)

    _install_menu()
