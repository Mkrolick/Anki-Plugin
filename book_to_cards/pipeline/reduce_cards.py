"""Reduce stage — collapse all aspects under a topic into 1-N atomic cards."""
from __future__ import annotations

from typing import Protocol

from .types import Card, Topic


class _LLM(Protocol):
    def chat_json(self, *, model: str, system: str, user: str, **kw) -> dict: ...


_SYSTEM_PROMPT = """\
You are writing Anki flashcards from study material on one topic. You will
be given:
- the topic name and its aliases,
- the FULL TEXT of every source page the topic appears on (use this for
  context: surrounding sentences, the author's name, what the chapter is
  arguing, definitions of terms used in the quotes),
- the extracted QUOTES that anchor each fact (these are the evidence; do
  not invent claims that aren't supported by them).

Produce 0-5 atomic flashcards.

Hard rules — break any of these and the card is unusable:

1. SELF-CONTAINED. The reader sees only the question, with no access to the
   source, the topic name, or any other card. They must understand exactly
   what is being asked.
   - BAD:  "What does the hypothesis state about doctors?"     (which hypothesis?)
   - BAD:  "How is this related to the quote?"                 (which quote?)
   - BAD:  "According to the text, what is the main idea?"     (which text?)
   - BAD:  "What is the function of the system?"               (which system?)
   - GOOD: "In Baudrillard's hypothesis about medicine, what role do doctors
            play in the simulation of disease?"
   - GOOD: "What does Baudrillard mean by 'the precession of simulacra'?"

   USE the topic name, the author's name (Baudrillard or whoever), and any
   specific anchor (a work title, a named concept) inside the QUESTION so
   the reader knows what they're being asked about. Never use phrases like
   "the text", "the source", "the quote", "the passage", "this", "the
   hypothesis" with no antecedent.

2. SUBSTANTIVE. Only emit a card if the fact is genuinely worth memorising —
   a concept, definition, named distinction, mechanism, claim, or argument
   that someone studying the work would want to retain.
   - SKIP passing mentions, decorative phrases, rhetorical questions in the
     source, and "an example of something is X" trivia that doesn't capture
     anything real.
   - SKIP if you cannot phrase a self-contained question without inventing
     context the quotes don't support.
   - It is BETTER to return an empty cards list than to produce filler.

3. ATOMIC. Each card tests exactly ONE fact. If the material warrants
   multiple cards, emit multiple cards.

4. GROUNDED. Every claim on the card must be supported by the provided quotes.
   Do not invent context. Do not paraphrase into something the quotes do not
   actually say.

5. FRONT is a question. BACK is the answer in 1-3 sentences. Both should
   read as natural prose, not as an extraction from a quote.

Respond with JSON: {"cards": [{"front": "...", "back": "...",
"source_quote_indices": [0, 2]}]}. To skip a topic entirely, return
{"cards": []}. No prose, no fences.
"""


def _build_user_prompt(topic: Topic, pages: dict | None = None) -> str:
    """Render the user message for one topic.

    `pages` is an optional {page_index: full_text} map. When supplied, the
    full text of every page touched by any aspect is included before the
    quote list so the model has surrounding context. Missing pages are
    silently skipped — the quotes still drive grounding.
    """
    pages = pages or {}
    lines: list[str] = []
    lines.append(f"Topic: {topic.canonical}")
    if topic.aliases and topic.aliases != [topic.canonical]:
        lines.append(f"Aliases: {', '.join(topic.aliases)}")
    lines.append("")

    referenced: list[int] = []
    seen: set[int] = set()
    for aspect in topic.aspects:
        for p in aspect.source_pages:
            try:
                pi = int(p)
            except (TypeError, ValueError):
                continue
            if pi not in seen and pi in pages:
                seen.add(pi)
                referenced.append(pi)
    referenced.sort()

    if referenced:
        lines.append("Source pages (use for context — NOT for evidence; only quotes count):")
        for pi in referenced:
            lines.append(f"[Page {pi}]")
            lines.append(pages[pi])
            lines.append("")

    lines.append("Quotes (cards must be grounded ONLY in these):")
    for i, aspect in enumerate(topic.aspects):
        pages_str = ", ".join(str(p) for p in aspect.source_pages)
        lines.append(f"  [{i}] \"{aspect.quote}\" (pp. {pages_str})")
    return "\n".join(lines)


def reduce_topic(
    topic: Topic,
    *,
    llm: _LLM,
    model: str = "gpt-4o-mini",
    pages: dict | None = None,
) -> list[Card]:
    response = llm.chat_json(
        model=model,
        system=_SYSTEM_PROMPT,
        user=_build_user_prompt(topic, pages),
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
