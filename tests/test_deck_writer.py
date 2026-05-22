"""Deck writer: create note type, insert cards, optional calibration hand-off.

Skips if the `anki` package isn't importable on the host Python. Anki ships
its own bundled Python; running these tests against a real Anki install
requires either using that interpreter or pip-installing the `anki` PyPI
package (which exists for headless use, but is a heavy dep we don't require).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

anki = pytest.importorskip("anki")


@pytest.fixture
def temp_collection(tmp_path):
    from anki.collection import Collection
    col = Collection(str(tmp_path / "test.anki2"))
    yield col
    col.close()


def test_ensures_note_type_and_deck(temp_collection):
    from book_to_cards.deck_writer import ensure_note_type, ensure_deck
    nt = ensure_note_type(temp_collection)
    assert nt["name"] == "Book Card"
    fields = {f["name"] for f in nt["flds"]}
    assert {"Front", "Back", "Topic", "Source"}.issubset(fields)

    did = ensure_deck(temp_collection, "Test Book")
    deck = temp_collection.decks.get(did)
    assert deck["name"] == "Test Book"


def test_inserts_card(temp_collection):
    from book_to_cards.deck_writer import ensure_note_type, ensure_deck, insert_card
    from book_to_cards.pipeline.types import Card
    nt = ensure_note_type(temp_collection)
    did = ensure_deck(temp_collection, "Test Book")
    card = Card(
        front="What is ATP?",
        back="The cell's energy currency.",
        topic="bioenergetics",
        source_pages=[3, 4],
        source_quotes=["ATP is produced via chemiosmosis."],
    )
    nid = insert_card(temp_collection, card, note_type=nt, deck_id=did, calibrate=False)
    note = temp_collection.get_note(nid)
    assert note["Front"] == "What is ATP?"
    assert "ATP is produced via chemiosmosis" in note["Source"]
    assert "p. 3" in note["Source"]
