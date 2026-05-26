"""Sliding-window iterator over pages, plus the LLM-driven map stage.

`chunk_iter` yields list[Page] windows; `run_map` calls an LLM client on
each chunk and yields one structured result dict per chunk (success or
isolated failure). The runner persists each yielded dict immediately so
a crash mid-loop loses at most one chunk's worth of in-flight work.
"""
from __future__ import annotations

from typing import Iterator

from .types import Page


def chunk_iter(pages: list[Page], *, size: int, overlap: int) -> Iterator[list[Page]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")
    if not pages:
        return
    step = size - overlap
    n = len(pages)
    if n <= size:
        yield list(pages)
        return
    start = 0
    last_end = -1
    while True:
        end = min(start + size, n)
        chunk = pages[start:end]
        if chunk:
            yield chunk
        last_end = end
        if end >= n:
            break
        start += step
    if last_end < n:
        yield pages[max(0, n - size):n]


_SYSTEM_PROMPT = """\
You are an expert at extracting study material from textbooks. For the
passage below, list every noteworthy fact, definition, mechanism, or claim
a student should remember. For each item return:
  - "text": a one-sentence distillation
  - "quote": the EXACT verbatim excerpt from the passage (do not edit)
  - "topic": a short topic name (1-4 words)
  - "source_pages": the page numbers (from the [Page N] markers) the excerpt came from

Respond with a single JSON object: {"aspects": [...]}. No prose, no fences.
"""


def _build_user_prompt(chunk: list[Page]) -> str:
    parts: list[str] = []
    for page in chunk:
        parts.append(f"[Page {page.index}]")
        parts.append(page.text)
    return "\n".join(parts)


def run_map(pages, *, llm, size: int, overlap: int, model: str | None = None):
    """Yield one dict per chunk.

    On success: {"chunk_index", "source_pages", "aspects": [...], "error": None}
    On failure: {"chunk_index", "source_pages", "aspects": [], "error": "msg"}
    """
    for i, chunk in enumerate(chunk_iter(pages, size=size, overlap=overlap)):
        source_pages = [p.index for p in chunk]
        try:
            response = llm.chat_json(
                model=model or "gpt-4o-mini",
                system=_SYSTEM_PROMPT,
                user=_build_user_prompt(chunk),
            )
            aspects = response.get("aspects") or []
            if not isinstance(aspects, list):
                raise RuntimeError(f"map returned non-list aspects: {type(aspects).__name__}")
            # Observation from real-book runs: the LLM is unreliable at echoing
            # back the absolute [Page N] markers we put in the prompt. It often
            # returns chunk-relative ints (0, 1, 2…) or even single small digits
            # unrelated to the actual page. That makes per-aspect attribution
            # untrustworthy, and the reduce stage's pages_map lookup would pull
            # wrong context. Drop whatever the LLM returned and replace with the
            # chunk's full page list — less precise per aspect, but guaranteed
            # to contain the real source page.
            for a in aspects:
                if isinstance(a, dict):
                    a["source_pages"] = list(source_pages)
            yield {
                "chunk_index": i,
                "source_pages": source_pages,
                "aspects": aspects,
                "error": None,
            }
        except Exception as e:
            yield {
                "chunk_index": i,
                "source_pages": source_pages,
                "aspects": [],
                "error": str(e),
            }
