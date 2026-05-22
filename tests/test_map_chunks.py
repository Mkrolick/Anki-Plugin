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
