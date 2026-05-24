"""Reduce stage: each Topic → 1-N Cards via the LLM."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _topic(canonical, aliases, aspects):
    from book_to_cards.pipeline.types import Aspect, Topic
    return Topic(canonical=canonical, aliases=aliases, aspects=[
        Aspect(text=a["text"], quote=a["quote"], topic=a["topic"], source_pages=a["pages"])
        for a in aspects
    ])


def test_reduce_emits_cards_with_source_quotes_resolved(fake_llm):
    from book_to_cards.pipeline.reduce_cards import reduce_topic
    canned = [{"cards": [
        {"front": "Q1", "back": "A1", "source_quote_indices": [0, 1]},
        {"front": "Q2", "back": "A2", "source_quote_indices": [1]},
    ]}]
    topic = _topic("photosynthesis", ["photosynthesis", "light reactions"], [
        {"text": "f0", "quote": "quote zero", "topic": "photosynthesis", "pages": [1]},
        {"text": "f1", "quote": "quote one", "topic": "light reactions", "pages": [3, 4]},
    ])
    cards = reduce_topic(topic, llm=fake_llm(canned))
    assert len(cards) == 2
    assert cards[0].source_quotes == ["quote zero", "quote one"]
    assert cards[0].source_pages == [1, 3, 4]
    assert cards[1].source_quotes == ["quote one"]


def test_reduce_carries_topic_name(fake_llm):
    from book_to_cards.pipeline.reduce_cards import reduce_topic
    topic = _topic("X", ["X"], [{"text": "f", "quote": "q", "topic": "X", "pages": [0]}])
    canned = [{"cards": [{"front": "Q", "back": "A", "source_quote_indices": [0]}]}]
    cards = reduce_topic(topic, llm=fake_llm(canned))
    assert cards[0].topic == "X"


def test_reduce_skips_invalid_indices(fake_llm):
    from book_to_cards.pipeline.reduce_cards import reduce_topic
    topic = _topic("X", ["X"], [{"text": "f", "quote": "q0", "topic": "X", "pages": [0]}])
    canned = [{"cards": [{"front": "Q", "back": "A", "source_quote_indices": [0, 5]}]}]
    cards = reduce_topic(topic, llm=fake_llm(canned))
    assert cards[0].source_quotes == ["q0"]


def test_reduce_includes_page_context_when_provided(fake_llm):
    """When the pages map is passed, every referenced page's full text is
    embedded in the user prompt before the quotes."""
    from book_to_cards.pipeline.reduce_cards import reduce_topic
    topic = _topic("X", ["X"], [
        {"text": "f", "quote": "short quote", "topic": "X", "pages": [3, 5]},
    ])
    pages_map = {
        3: "Full text of page 3 with surrounding sentences and more context.",
        5: "Full text of page 5 explaining the broader argument.",
        7: "Unrelated page — should not appear.",
    }
    canned = [{"cards": [{"front": "Q", "back": "A", "source_quote_indices": [0]}]}]
    llm = fake_llm(canned)
    reduce_topic(topic, llm=llm, pages=pages_map)
    user_msg = llm.calls[0]["user"]
    assert "[Page 3]" in user_msg
    assert "Full text of page 3" in user_msg
    assert "[Page 5]" in user_msg
    assert "Unrelated page" not in user_msg


def test_reduce_works_without_pages_map(fake_llm):
    """Backwards-compat: omitting the pages map still produces a valid prompt."""
    from book_to_cards.pipeline.reduce_cards import reduce_topic
    topic = _topic("X", ["X"], [{"text": "f", "quote": "q", "topic": "X", "pages": [1]}])
    canned = [{"cards": [{"front": "Q", "back": "A", "source_quote_indices": [0]}]}]
    llm = fake_llm(canned)
    cards = reduce_topic(topic, llm=llm)
    assert len(cards) == 1
    assert "Source pages" not in llm.calls[0]["user"]  # section omitted when no map
