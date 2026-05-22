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
