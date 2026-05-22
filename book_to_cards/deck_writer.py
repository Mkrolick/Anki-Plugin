"""Anki deck and note operations for book_to_cards.

Idempotent helpers: ensure_note_type, ensure_deck, insert_card. The note
type is created with the field layout smart_grader expects so calibration
just works when the grader add-on is installed.
"""
from __future__ import annotations

import html
import importlib.util

from .pipeline.types import Card

NOTE_TYPE_NAME = "Book Card"
USER_FIELDS = ["Front", "Back", "Topic", "Source"]
GRADER_FIELDS = [
    "Keywords",
    "_grader_threshold",
    "_grader_ref_embedding",
    "_grader_calibration",
]


def _smart_grader_available() -> bool:
    return importlib.util.find_spec("smart_grader") is not None


def ensure_note_type(col) -> dict:
    """Create the Book Card note type if missing; idempotent."""
    mm = col.models
    existing = mm.by_name(NOTE_TYPE_NAME)
    if existing is not None:
        existing_names = {f["name"] for f in existing["flds"]}
        for name in USER_FIELDS + GRADER_FIELDS:
            if name not in existing_names:
                mm.add_field(existing, mm.new_field(name))
        mm.save(existing)
        return existing

    nt = mm.new(NOTE_TYPE_NAME)
    for name in USER_FIELDS + GRADER_FIELDS:
        mm.add_field(nt, mm.new_field(name))
    tmpl = mm.new_template("Card 1")
    tmpl["qfmt"] = "{{Front}}"
    tmpl["afmt"] = "{{FrontSide}}<hr id=answer>{{Back}}"
    mm.add_template(nt, tmpl)
    mm.add(nt)
    return mm.by_name(NOTE_TYPE_NAME)


def ensure_deck(col, name: str) -> int:
    return col.decks.id(name)


def _format_source(card: Card) -> str:
    if not card.source_quotes:
        return ""
    lines: list[str] = []
    pages = card.source_pages or [None] * len(card.source_quotes)
    # Pad pages to match quotes if shorter.
    if len(pages) < len(card.source_quotes):
        pages = list(pages) + [None] * (len(card.source_quotes) - len(pages))
    for q, p in zip(card.source_quotes, pages):
        page_tag = f"p. {p} — " if p is not None else ""
        lines.append(f"{page_tag}“{html.escape(q)}”")
    return "<br>".join(lines)


def insert_card(col, card: Card, *, note_type: dict, deck_id: int, calibrate: bool = True) -> int:
    """Create a note in the chosen deck. Optionally hand off to smart_grader."""
    note = col.new_note(note_type)
    note["Front"] = card.front
    note["Back"] = card.back
    note["Topic"] = card.topic
    note["Source"] = _format_source(card)
    col.add_note(note, deck_id)

    if calibrate and _smart_grader_available():
        try:
            from smart_grader.api import calibrate_note  # type: ignore
            calibrate_note(note)
        except Exception:
            # Calibration is best-effort; if it fails, the card still exists.
            pass

    return note.id
