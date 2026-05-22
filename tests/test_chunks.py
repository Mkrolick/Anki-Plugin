"""Chunk iterator: sliding window of pages with overlap."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _pages(n: int):
    from book_to_cards.pipeline.types import Page
    return [Page(index=i, text=f"page {i} content") for i in range(n)]


def test_single_chunk_when_book_shorter_than_window():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    chunks = list(chunk_iter(_pages(3), size=5, overlap=1))
    assert len(chunks) == 1
    assert [p.index for p in chunks[0]] == [0, 1, 2]


def test_exact_window_one_chunk():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    chunks = list(chunk_iter(_pages(5), size=5, overlap=1))
    assert len(chunks) == 1
    assert [p.index for p in chunks[0]] == [0, 1, 2, 3, 4]


def test_overlap_step_correct():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    chunks = list(chunk_iter(_pages(13), size=5, overlap=1))
    assert len(chunks) == 3
    assert [p.index for p in chunks[0]] == [0, 1, 2, 3, 4]
    assert [p.index for p in chunks[1]] == [4, 5, 6, 7, 8]
    assert [p.index for p in chunks[2]] == [8, 9, 10, 11, 12]


def test_every_page_appears_at_least_once():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    pages = _pages(17)
    seen: set[int] = set()
    for chunk in chunk_iter(pages, size=5, overlap=1):
        seen.update(p.index for p in chunk)
    assert seen == set(range(17))


def test_tail_handles_uneven_book():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    pages = _pages(11)
    chunks = list(chunk_iter(pages, size=5, overlap=1))
    flat = [p.index for c in chunks for p in c]
    assert set(flat) == set(range(11))
    assert chunks[-1][-1].index == 10
