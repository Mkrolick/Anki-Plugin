"""Book to Cards for Anki — PDF ingest → auto-generated flashcards.

Loaded by Anki at startup. Registers the Tools menu entry; full
wiring lands in Task 16.
"""
from __future__ import annotations

from aqt import mw
from aqt.qt import QAction


def _on_generate():
    from aqt.utils import showInfo
    showInfo("Book to Cards: not yet implemented.")


def _install_menu():
    action = QAction("Generate cards from PDF…", mw)
    action.triggered.connect(_on_generate)
    mw.form.menuTools.addAction(action)


_install_menu()
