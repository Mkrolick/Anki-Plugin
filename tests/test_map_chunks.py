"""Map stage: chunks → aspects via the LLM client."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _pages(n):
    from book_to_cards.pipeline.types import Page
    return [Page(index=i, text=f"page {i} content about topic_{i % 3}") for i in range(n)]


def test_map_emits_one_result_per_chunk(fake_llm):
    from book_to_cards.pipeline.map_chunks import run_map
    canned = [
        {"aspects": [{"text": "fact A", "quote": "qA", "topic": "tA", "source_pages": [0]}]},
        {"aspects": [{"text": "fact B", "quote": "qB", "topic": "tB", "source_pages": [4]}]},
        {"aspects": [{"text": "fact C", "quote": "qC", "topic": "tC", "source_pages": [8]}]},
    ]
    llm = fake_llm(canned)
    results = list(run_map(_pages(13), llm=llm, size=5, overlap=1))
    assert len(results) == 3
    assert llm.calls[0]["model"]
    assert "page 0 content" in llm.calls[0]["user"]


def test_map_attaches_source_page_indices(fake_llm):
    from book_to_cards.pipeline.map_chunks import run_map
    canned = [
        {"aspects": [{"text": "x", "quote": "q", "topic": "t", "source_pages": [0, 1]}]},
    ]
    llm = fake_llm(canned)
    [chunk_result] = list(run_map(_pages(3), llm=llm, size=5, overlap=1))
    assert chunk_result["chunk_index"] == 0
    assert chunk_result["source_pages"] == [0, 1, 2]
    assert chunk_result["aspects"][0]["topic"] == "t"


def test_map_overrides_aspect_source_pages_with_chunk_pages(fake_llm):
    """The LLM is unreliable at echoing back absolute page markers — it often
    returns chunk-relative or random small ints. run_map must overwrite each
    aspect's source_pages with the chunk's actual page indices so reduce's
    pages_map lookup hits the right pages.
    """
    from book_to_cards.pipeline.map_chunks import run_map
    from book_to_cards.pipeline.types import Page

    # 6-page book; chunk_iter with size=5, overlap=1 yields chunks
    # [0..4] and [4..5]. We'll only verify the first chunk for clarity.
    pages = [Page(index=i, text=f"page {i}") for i in range(6)]

    # LLM returns wildly wrong source_pages — both relative ints AND
    # an out-of-range page that doesn't exist in this chunk.
    canned = [
        {"aspects": [
            {"text": "fact A", "quote": "q", "topic": "t",
             "source_pages": [0]},          # LLM says page 0
            {"text": "fact B", "quote": "q", "topic": "t",
             "source_pages": [1]},          # LLM says page 1
            {"text": "fact C", "quote": "q", "topic": "t",
             "source_pages": [42]},         # LLM hallucinated page 42
        ]},
        {"aspects": []},
    ]
    llm = fake_llm(canned)
    results = list(run_map(pages, llm=llm, size=5, overlap=1))

    first = results[0]
    expected_pages = [0, 1, 2, 3, 4]
    assert first["source_pages"] == expected_pages
    # Every aspect's source_pages must now equal the chunk's page list,
    # regardless of what the LLM returned.
    for asp in first["aspects"]:
        assert asp["source_pages"] == expected_pages, (
            f"aspect.source_pages should be overridden to chunk pages; "
            f"got {asp['source_pages']}"
        )


def test_map_override_preserves_chunk_membership_across_chunks(fake_llm):
    """When chunks have different page ranges, each chunk's aspects get
    their own chunk's pages — they don't leak across."""
    from book_to_cards.pipeline.map_chunks import run_map
    from book_to_cards.pipeline.types import Page

    pages = [Page(index=i, text=f"page {i}") for i in range(13)]
    # chunk_iter(size=5, overlap=1) yields [0..4], [4..8], [8..12]
    canned = [
        {"aspects": [{"text": "x", "quote": "q", "topic": "t", "source_pages": [0]}]},
        {"aspects": [{"text": "x", "quote": "q", "topic": "t", "source_pages": [1]}]},
        {"aspects": [{"text": "x", "quote": "q", "topic": "t", "source_pages": [99]}]},
    ]
    llm = fake_llm(canned)
    results = list(run_map(pages, llm=llm, size=5, overlap=1))

    assert results[0]["aspects"][0]["source_pages"] == [0, 1, 2, 3, 4]
    assert results[1]["aspects"][0]["source_pages"] == [4, 5, 6, 7, 8]
    assert results[2]["aspects"][0]["source_pages"] == [8, 9, 10, 11, 12]


def test_map_continues_on_chunk_failure():
    from book_to_cards.pipeline.map_chunks import run_map

    class FlakyLLM:
        def __init__(self):
            self.n = 0
            self.calls = []
        def chat_json(self, **kw):
            self.calls.append(kw)
            self.n += 1
            if self.n == 2:
                raise RuntimeError("boom")
            return {"aspects": [{"text": "ok", "quote": "q", "topic": "t",
                                  "source_pages": [self.n]}]}

    llm = FlakyLLM()
    results = list(run_map(_pages(13), llm=llm, size=5, overlap=1))
    good = [r for r in results if r.get("error") is None]
    errors = [r for r in results if r.get("error") is not None]
    assert len(good) == 2
    assert len(errors) == 1
    assert errors[0]["chunk_index"] == 1
