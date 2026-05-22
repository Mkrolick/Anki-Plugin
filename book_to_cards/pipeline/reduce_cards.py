"""Reduce stage — collapse all aspects under a topic into 1-N atomic cards."""
from __future__ import annotations

from typing import Protocol

from .types import Card, Topic


class _LLM(Protocol):
    def chat_json(self, *, model: str, system: str, user: str, **kw) -> dict: ...


_SYSTEM_PROMPT = """\
You are writing Anki flashcards from study material on one topic. You will
be given the topic name, its aliases, and several extracted facts (each
with a verbatim source quote). Produce 1-5 atomic flashcards.

Rules:
- Each card tests EXACTLY ONE fact. If a topic has multiple distinct facts,
  emit multiple cards.
- Ground every card in the provided quotes. Do not introduce information
  that is not supported by them.
- Front is a question. Back is the answer in 1-3 sentences.
- Don't include the topic name in the front unless it's the natural way to
  phrase the question.

Respond with JSON: {"cards": [{"front": "...", "back": "...",
"source_quote_indices": [0, 2]}]}. No prose, no fences.
"""


def _build_user_prompt(topic: Topic) -> str:
    lines: list[str] = []
    lines.append(f"Topic: {topic.canonical}")
    if topic.aliases and topic.aliases != [topic.canonical]:
        lines.append(f"Aliases: {', '.join(topic.aliases)}")
    lines.append("")
    lines.append("Quotes:")
    for i, aspect in enumerate(topic.aspects):
        pages = ", ".join(str(p) for p in aspect.source_pages)
        lines.append(f"  [{i}] \"{aspect.quote}\" (pp. {pages})")
    return "\n".join(lines)


def reduce_topic(topic: Topic, *, llm: _LLM, model: str = "gpt-4o-mini") -> list[Card]:
    response = llm.chat_json(
        model=model,
        system=_SYSTEM_PROMPT,
        user=_build_user_prompt(topic),
    )
    raw_cards = response.get("cards") or []
    out: list[Card] = []
    for raw in raw_cards:
        front = str(raw.get("front", "")).strip()
        back = str(raw.get("back", "")).strip()
        if not front or not back:
            continue
        indices = raw.get("source_quote_indices") or []
        quotes: list[str] = []
        pages: set[int] = set()
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(topic.aspects):
                quotes.append(topic.aspects[idx].quote)
                pages.update(topic.aspects[idx].source_pages)
        out.append(Card(
            front=front,
            back=back,
            topic=topic.canonical,
            source_pages=sorted(pages),
            source_quotes=quotes,
        ))
    return out
