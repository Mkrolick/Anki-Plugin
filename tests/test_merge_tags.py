"""Merge stage: topic strings → canonical clusters via cosine clustering."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _unit_vector(angle_rad: float, dim: int = 4) -> list[float]:
    v = [math.cos(angle_rad), math.sin(angle_rad)] + [0.0] * (dim - 2)
    return v


def _aspects_with_topics(topics):
    from book_to_cards.pipeline.types import Aspect
    return [Aspect(text=t, quote=t, topic=t, source_pages=[0]) for t in topics]


def test_close_tags_merge(fake_embedder):
    from book_to_cards.pipeline.merge_tags import merge_topics
    vecs = {
        "chemiosmosis": _unit_vector(0.00),
        "electron transport chain": _unit_vector(0.05),
        "oxidative phosphorylation": _unit_vector(0.10),
        "photosynthesis": _unit_vector(2.50),
    }
    aspects = _aspects_with_topics(list(vecs.keys()))
    topics = merge_topics(aspects, embedder=fake_embedder(vecs), distance_threshold=0.25)
    assert len(topics) == 2
    bio_cluster = next(t for t in topics if "photosynthesis" in t.aliases)
    chem_cluster = next(t for t in topics if t is not bio_cluster)
    assert len(chem_cluster.aliases) == 3
    assert len(bio_cluster.aliases) == 1


def test_each_aspect_lands_in_exactly_one_cluster(fake_embedder):
    from book_to_cards.pipeline.merge_tags import merge_topics
    vecs = {
        "a": _unit_vector(0.0),
        "b": _unit_vector(0.05),
        "c": _unit_vector(3.0),
    }
    aspects = _aspects_with_topics(["a", "b", "c"])
    topics = merge_topics(aspects, embedder=fake_embedder(vecs), distance_threshold=0.25)
    seen: set[str] = set()
    for t in topics:
        for a in t.aspects:
            assert a.topic not in seen
            seen.add(a.topic)
    assert seen == {"a", "b", "c"}


def test_unembeddable_tag_goes_to_uncategorized():
    from book_to_cards.pipeline.merge_tags import merge_topics

    class FlakyEmbedder:
        def __init__(self):
            self.calls = []
        def embed(self, text):
            self.calls.append(text)
            if text == "broken":
                raise RuntimeError("nope")
            return _unit_vector(0.0)

    aspects = _aspects_with_topics(["working", "broken"])
    topics = merge_topics(aspects, embedder=FlakyEmbedder(), distance_threshold=0.25)
    canonicals = {t.canonical for t in topics}
    assert "uncategorized" in canonicals
    uncat = next(t for t in topics if t.canonical == "uncategorized")
    assert [a.topic for a in uncat.aspects] == ["broken"]
