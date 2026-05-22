"""PDF extraction smoke tests."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "book_to_cards" / "_vendor"))


def test_extracts_pages_with_known_text(small_pdf_path):
    from book_to_cards.pdf_text import extract_pages
    pages = extract_pages(str(small_pdf_path))
    assert len(pages) == 5
    assert pages[0].index == 0
    assert "Photosynthesis" in pages[0].text


def test_pages_text_is_whitespace_normalized(small_pdf_path):
    from book_to_cards.pdf_text import extract_pages
    pages = extract_pages(str(small_pdf_path))
    for p in pages:
        assert re.search(r"\s{2,}", p.text) is None, f"page {p.index}: {p.text!r}"


def test_useful_page_threshold():
    from book_to_cards.pdf_text import _is_useful_page, _USEFUL_PAGE_MIN_CHARS
    # Legit page with one substantive sentence: kept
    assert _is_useful_page("Photosynthesis converts light energy.") is True
    # Whitespace / running header noise: dropped
    assert _is_useful_page("   ") is False
    assert _is_useful_page("Ch. 4 | 47") is False
    # Boundary check
    assert _is_useful_page("x" * (_USEFUL_PAGE_MIN_CHARS - 1)) is False
    assert _is_useful_page("x" * _USEFUL_PAGE_MIN_CHARS) is True
