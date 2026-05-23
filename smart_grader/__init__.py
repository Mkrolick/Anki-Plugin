"""
Smart Grader for Anki
=====================

Adds a typed-answer flow to cards that have a `Keywords` field.

Pipeline on submit:
    1. Strict AND keyword check (case-insensitive, word-boundary).
    2. Cosine similarity between user-answer embedding and reference embedding.
    3. Compare against per-card calibrated threshold (stored in `_grader_threshold`).
    4. Show pass/fail + diff; user still picks Again/Hard/Good/Easy.

Calibration (Tools menu):
    For each selected note, ask GPT for ~12 "good" paraphrases and ~6 "bad"
    near-misses, embed everything, set threshold = max(min(good_sims),
    max(bad_sims) + epsilon). Store mean/stddev for diagnostics.
"""

from __future__ import annotations

import json
from typing import Optional

try:
    from aqt import gui_hooks, mw
    from aqt.qt import QAction, QMessageBox
    from aqt.utils import showInfo, tooltip
    _AQT_AVAILABLE = True
except ImportError:
    _AQT_AVAILABLE = False

from .grader import evaluate_answer
from .calibration import calibrate_selected_notes
from .config import get_config
from .ui import build_input_html, INJECT_JS

# Field names we look for on the note. Cards without `Keywords` are ignored
# (the add-on becomes a no-op so it doesn't interfere with normal decks).
KEYWORDS_FIELD = "Keywords"
ANSWER_FIELD_CANDIDATES = ("Back", "Answer", "Definition")
THRESHOLD_FIELD = "_grader_threshold"
REF_EMBEDDING_FIELD = "_grader_ref_embedding"


def _find_answer_field(note) -> Optional[str]:
    """Return the first present answer field on the note, or None."""
    for name in ANSWER_FIELD_CANDIDATES:
        if name in note:
            return name
    return None


def on_card_will_show(html: str, card, context: str) -> str:
    """Inject the typing UI on the question side of eligible cards.

    Only the HTML island (textarea, button, payload data-script tag) is
    appended here; the JS handler is wired up by on_reviewer_did_show_question
    via mw.reviewer.web.eval(...) because modern Anki strips <script> tags
    from card HTML.
    """
    if not context.endswith("Question"):
        return html

    note = card.note()
    if KEYWORDS_FIELD not in note:
        return html

    answer_field = _find_answer_field(note)
    if answer_field is None:
        return html

    payload = {
        "note_id": note.id,
        "reference": note[answer_field],
        "keywords": [k.strip() for k in note[KEYWORDS_FIELD].split(",") if k.strip()],
        "threshold": float(note[THRESHOLD_FIELD]) if THRESHOLD_FIELD in note and note[THRESHOLD_FIELD] else 0.82,
    }
    return html + build_input_html(payload)


def on_reviewer_did_show_question(card):
    """Inject the click + keydown handlers into the reviewer webview."""
    note = card.note()
    if KEYWORDS_FIELD not in note:
        return
    if _find_answer_field(note) is None:
        return
    mw.reviewer.web.eval(INJECT_JS)


def on_js_message(handled, message: str, context):
    """Handle pycmd('smartgrader:...') messages coming from the card webview."""
    if not message.startswith("smartgrader:"):
        return handled

    try:
        payload = json.loads(message[len("smartgrader:"):])
    except json.JSONDecodeError:
        return (True, None)

    # Pull the pre-computed reference embedding off the note (set during
    # calibration) so we don't re-embed the reference on every submit.
    ref_embedding = None
    try:
        note = mw.col.get_note(int(payload["note_id"]))
        if REF_EMBEDDING_FIELD in note and note[REF_EMBEDDING_FIELD]:
            ref_embedding = json.loads(note[REF_EMBEDDING_FIELD])
    except Exception:
        ref_embedding = None

    result = evaluate_answer(
        user_answer=payload["user_answer"],
        reference=payload["reference"],
        keywords=payload["keywords"],
        threshold=payload["threshold"],
        reference_embedding=ref_embedding,
    )

    # Push the result back into the webview as JSON; the JS side renders it.
    js = f"window.smartGraderShowResult({json.dumps(result)});"
    mw.reviewer.web.eval(js)
    return (True, None)


def on_calibrate_action():
    """Menu entry: calibrate currently-selected notes in the browser."""
    from aqt.browser import Browser

    # Only meaningful when the browser is open with a selection.
    browser = next((w for w in mw.app.topLevelWidgets() if isinstance(w, Browser) and w.isVisible()), None)
    if browser is None:
        showInfo("Open the Browse window and select notes to calibrate first.")
        return

    nids = browser.selectedNotes()
    if not nids:
        showInfo("No notes selected.")
        return

    if not get_config().get("openai_api_key"):
        showInfo("Set your OpenAI API key in Tools → Add-ons → Config first.")
        return

    n_done, n_skipped = calibrate_selected_notes(nids, mw.col)
    tooltip(f"Calibrated {n_done} note(s); skipped {n_skipped}.")


def _install_menu():
    action = QAction("Smart Grader: Calibrate selected notes", mw)
    action.triggered.connect(on_calibrate_action)
    mw.form.menuTools.addAction(action)


# Register hooks at import time. Anki imports __init__.py once on startup.
# Headless tests can still import the package; only the Anki-side wiring is
# guarded behind aqt's presence.
if _AQT_AVAILABLE:
    gui_hooks.card_will_show.append(on_card_will_show)
    gui_hooks.reviewer_did_show_question.append(on_reviewer_did_show_question)
    gui_hooks.webview_did_receive_js_message.append(on_js_message)
    _install_menu()
