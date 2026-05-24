"""OpenAI client wrapper for book_to_cards.

Vendored from smart_grader so the two add-ons stay independently
installable. Two responsibilities:
    1. embed(text) -> list[float], cached on disk by SHA256.
    2. chat_json(system, user, model) -> dict, JSON-mode response.

Also tracks cumulative (input_tokens, output_tokens) so the pipeline
runner can enforce a cost ceiling.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request

from .config import get_config


_CACHE_PATH = os.path.join(os.path.dirname(__file__), "embedding_cache.sqlite")

# Per 1M tokens (USD), as of 2026-01.
_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


class CostMeter:
    """Thread-safe cumulative cost tracker."""
    def __init__(self):
        self._lock = threading.Lock()
        self._usd = 0.0
        self._tokens: dict[str, dict[str, int]] = {}

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        prices = _PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        with self._lock:
            self._usd += cost
            mt = self._tokens.setdefault(model, {"input": 0, "output": 0})
            mt["input"] += input_tokens
            mt["output"] += output_tokens

    @property
    def usd(self) -> float:
        with self._lock:
            return self._usd

    def snapshot(self) -> dict:
        with self._lock:
            return {"usd": self._usd, "by_model": {k: dict(v) for k, v in self._tokens.items()}}


def _cache_conn() -> sqlite3.Connection:
    # 10s busy_timeout so the cache survives parallel calibration workers
    # opening fresh connections concurrently.
    conn = sqlite3.connect(_CACHE_PATH, timeout=10.0)
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "  key TEXT PRIMARY KEY,"
        "  model TEXT NOT NULL,"
        "  vector TEXT NOT NULL"
        ")"
    )
    return conn


def _cache_key(text: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _http_post_json(url: str, body: dict, api_key: str, timeout: int = 60) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {body_txt}") from e


class OpenAIClient:
    def __init__(self, api_key: str | None = None, meter: CostMeter | None = None):
        cfg = get_config()
        self._api_key = api_key or cfg["openai_api_key"]
        self.meter = meter or CostMeter()
        if not self._api_key:
            raise RuntimeError("OpenAI API key not configured.")

    def embed(self, text: str, *, model: str | None = None) -> list[float]:
        """Single-input wrapper around batch_embed."""
        return self.batch_embed([text], model=model)[0]

    def batch_embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Embed `texts` in (at most) one HTTP call.

        Cache-aware: already-known vectors come from SQLite; only the uncached
        subset hits the network. Vectors are returned in input order.
        """
        if not texts:
            return []
        cfg = get_config()
        m = model or cfg["embedding_model"]

        results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        conn = _cache_conn()
        try:
            for i, text in enumerate(texts):
                key = _cache_key(text, m)
                row = conn.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
                if row is not None:
                    results[i] = json.loads(row[0])
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            if uncached_texts:
                resp = _http_post_json(
                    "https://api.openai.com/v1/embeddings",
                    {"model": m, "input": uncached_texts},
                    self._api_key,
                )
                usage = resp.get("usage", {})
                self.meter.record(m, usage.get("prompt_tokens", 0), 0)
                for batch_idx, orig_idx in enumerate(uncached_indices):
                    vec = resp["data"][batch_idx]["embedding"]
                    results[orig_idx] = vec
                    key = _cache_key(uncached_texts[batch_idx], m)
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings(key, model, vector) VALUES (?, ?, ?)",
                        (key, m, json.dumps(vec)),
                    )
                conn.commit()
        finally:
            conn.close()

        return [results[i] for i in range(len(texts))]

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> dict:
        cfg = get_config()
        m = model or cfg["chat_model"]
        resp = _http_post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": m,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "temperature": temperature,
            },
            self._api_key,
        )
        usage = resp.get("usage", {})
        self.meter.record(m, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content)
