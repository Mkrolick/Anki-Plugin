"""
OpenAI client wrapper.

Two responsibilities:
    1. embed(text) -> list[float], with on-disk SHA256 cache.
    2. generate_paraphrases(reference, keywords) -> {"good": [...], "bad": [...]}

We use the Responses API JSON-mode flow rather than function calling — simpler
and the schema we need is trivial. If the API returns malformed JSON we raise;
the caller logs and skips that note rather than corrupting calibration.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.request
import urllib.error
from typing import Optional

from .config import get_config


_CACHE_PATH = os.path.join(os.path.dirname(__file__), "embedding_cache.sqlite")


def _cache_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_CACHE_PATH)
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


def _http_post_json(url: str, body: dict, api_key: str, timeout: int = 30) -> dict:
    """Minimal stdlib POST so the add-on has no pip dependencies."""
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
        # Surface the API's error body — it usually says exactly what's wrong.
        body_txt = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {body_txt}") from e


def embed(text: str) -> list[float]:
    """Return the embedding for `text`, caching by (model, text) hash.

    Single-input wrapper around batch_embed; kept for backwards compatibility.
    """
    return batch_embed([text])[0]


def batch_embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in (at most) one HTTP call.

    Cache-aware: any texts whose vectors are already on disk are served from
    SQLite; only the uncached subset hits the network. Returns vectors in
    the same order as the input list.

    Empty list returns []; the OpenAI API rejects empty inputs.
    """
    if not texts:
        return []
    cfg = get_config()
    model = cfg["embedding_model"]
    api_key = cfg["openai_api_key"]
    if not api_key:
        raise RuntimeError("OpenAI API key not configured.")

    results: dict[int, list[float]] = {}
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    conn = _cache_conn()
    try:
        for i, text in enumerate(texts):
            key = _cache_key(text, model)
            row = conn.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
            if row is not None:
                results[i] = json.loads(row[0])
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            resp = _http_post_json(
                "https://api.openai.com/v1/embeddings",
                {"model": model, "input": uncached_texts},
                api_key,
            )
            for batch_idx, orig_idx in enumerate(uncached_indices):
                vec = resp["data"][batch_idx]["embedding"]
                results[orig_idx] = vec
                key = _cache_key(uncached_texts[batch_idx], model)
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings(key, model, vector) VALUES (?, ?, ?)",
                    (key, model, json.dumps(vec)),
                )
            conn.commit()
    finally:
        conn.close()

    return [results[i] for i in range(len(texts))]


_KEYWORDS_PROMPT = """\
You are helping configure a flashcard grader. Given a question and its
reference answer, list the small set of terms a correct answer MUST contain
to be considered right. Pick concrete nouns / named concepts / numbers — the
load-bearing words. Skip filler ("the", "and", generic verbs) and skip
synonyms of words already in the list. Aim for 3-6 keywords; fewer is fine
if the answer is short.

Respond with a single JSON object: {{"keywords": ["...", "..."]}}. No prose,
no markdown fences.

Question:
\"\"\"{question}\"\"\"

Reference answer:
\"\"\"{answer}\"\"\"
"""


def generate_keywords(question: str, answer: str) -> list[str]:
    """Ask the chat model for the required keywords for this card."""
    cfg = get_config()
    api_key = cfg["openai_api_key"]
    if not api_key:
        raise RuntimeError("OpenAI API key not configured.")

    prompt = _KEYWORDS_PROMPT.format(question=question, answer=answer)
    resp = _http_post_json(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": cfg["chat_model"],
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
        api_key,
        timeout=60,
    )
    content = resp["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    kws = parsed.get("keywords") or []
    if not isinstance(kws, list):
        raise RuntimeError(f"Malformed keywords response: {content[:200]}")
    return [str(k).strip() for k in kws if str(k).strip()]


_PARAPHRASE_PROMPT = """\
You are helping calibrate a flashcard grader. Given a reference answer and a
list of required keywords, produce two sets of rewordings:

  good: {n_good} variants that a knowledgeable student might give as their own
        wording. They MUST preserve the full meaning and MUST contain every
        required keyword (in any form). Vary sentence structure and vocabulary.

  bad:  {n_bad} variants that sound plausible but are semantically wrong or
        missing a key idea — e.g. swap a critical noun, invert a relationship,
        omit a defining property, or describe a related-but-different concept.
        These may or may not contain the keywords.

Respond with a single JSON object: {{"good": [...], "bad": [...]}}. No prose,
no markdown fences.

Reference answer:
\"\"\"{reference}\"\"\"

Required keywords: {keywords}
"""


def generate_paraphrases(reference: str, keywords: list[str]) -> dict:
    """Ask the chat model for good/bad paraphrases. Returns {"good": [...], "bad": [...]}."""
    cfg = get_config()
    api_key = cfg["openai_api_key"]
    if not api_key:
        raise RuntimeError("OpenAI API key not configured.")

    prompt = _PARAPHRASE_PROMPT.format(
        n_good=cfg["n_good_paraphrases"],
        n_bad=cfg["n_bad_paraphrases"],
        reference=reference,
        keywords=", ".join(keywords) if keywords else "(none)",
    )
    resp = _http_post_json(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": cfg["chat_model"],
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
        },
        api_key,
        timeout=60,
    )
    content = resp["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    # Defensive: ensure both keys present and lists.
    good = parsed.get("good") or []
    bad = parsed.get("bad") or []
    if not isinstance(good, list) or not isinstance(bad, list):
        raise RuntimeError(f"Malformed paraphrase response: {content[:200]}")
    return {"good": [str(x) for x in good], "bad": [str(x) for x in bad]}
