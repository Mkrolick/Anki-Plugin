"""batch_embed: single HTTP call for many inputs, cache-aware ordering."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_smart_grader_batch_embed_orders_cached_and_fresh(tmp_path, monkeypatch):
    """A mix of cached and uncached texts should return in input order."""
    cache_path = tmp_path / "embedding_cache.sqlite"
    monkeypatch.setattr("smart_grader.openai_client._CACHE_PATH", str(cache_path))

    # Pre-seed the cache with one known vector.
    from smart_grader.openai_client import _cache_conn, _cache_key
    model = "text-embedding-3-small"
    conn = _cache_conn()
    cached_vec = [0.1] * 4
    conn.execute(
        "INSERT INTO embeddings(key, model, vector) VALUES (?, ?, ?)",
        (_cache_key("cached-text", model), model, json.dumps(cached_vec)),
    )
    conn.commit()
    conn.close()

    fresh_vec = [0.9] * 4

    def fake_post(url, body, api_key, **kw):
        # Only uncached texts should be in the request.
        assert body["input"] == ["fresh-text"]
        return {"data": [{"embedding": fresh_vec}]}

    monkeypatch.setattr("smart_grader.openai_client._http_post_json", fake_post)
    monkeypatch.setattr("smart_grader.openai_client.get_config",
                        lambda: {"embedding_model": model, "openai_api_key": "k"})

    from smart_grader.openai_client import batch_embed
    out = batch_embed(["cached-text", "fresh-text", "cached-text"])
    assert out[0] == cached_vec
    assert out[1] == fresh_vec
    assert out[2] == cached_vec


def test_smart_grader_batch_embed_empty_input_no_request(monkeypatch):
    """Empty input must not hit the network."""
    called = []
    monkeypatch.setattr("smart_grader.openai_client._http_post_json",
                        lambda *a, **kw: called.append(1) or {})
    monkeypatch.setattr("smart_grader.openai_client.get_config",
                        lambda: {"embedding_model": "m", "openai_api_key": "k"})
    from smart_grader.openai_client import batch_embed
    assert batch_embed([]) == []
    assert not called


def test_book_to_cards_batch_embed_records_cost(monkeypatch, tmp_path):
    """book_to_cards' batch_embed must update its CostMeter from the API usage."""
    monkeypatch.setattr("book_to_cards.openai_client._CACHE_PATH",
                        str(tmp_path / "ec.sqlite"))
    monkeypatch.setattr("book_to_cards.openai_client.get_config",
                        lambda: {"embedding_model": "text-embedding-3-small",
                                  "openai_api_key": "k"})

    fresh_vec = [0.5] * 4
    def fake_post(url, body, api_key, **kw):
        return {
            "data": [{"embedding": fresh_vec} for _ in body["input"]],
            "usage": {"prompt_tokens": 1234},
        }
    monkeypatch.setattr("book_to_cards.openai_client._http_post_json", fake_post)

    from book_to_cards.openai_client import OpenAIClient, CostMeter
    client = OpenAIClient(api_key="k", meter=CostMeter())
    vecs = client.batch_embed(["a", "b", "c"])
    assert len(vecs) == 3 and all(v == fresh_vec for v in vecs)
    assert client.meter.usd > 0
