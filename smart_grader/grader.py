"""
Grading pipeline.

evaluate_answer() returns a dict the JS frontend renders into the result panel.
It's intentionally stateless — all per-card config (keywords, threshold,
reference text) comes in via arguments.
"""

from __future__ import annotations

import difflib
import math
import re

from .openai_client import embed


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace. Used for keyword matching only."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def check_keywords(user_answer: str, keywords: list[str]) -> dict:
    """
    Strict AND: every keyword must appear as a whole word (case-insensitive).
    Returns {"ok": bool, "missing": [...], "found": [...]}.
    """
    normalized = _normalize(user_answer)
    found, missing = [], []
    for kw in keywords:
        # \b around the keyword; escape regex metacharacters inside the keyword.
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, normalized):
            found.append(kw)
        else:
            missing.append(kw)
    return {"ok": not missing, "missing": missing, "found": found}


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Vectors are non-empty and same length by construction."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _diff_html(user_answer: str, reference: str) -> str:
    """Inline word-level diff. Green = correct, red strike = extra, gray = missing."""
    matcher = difflib.SequenceMatcher(a=user_answer.split(), b=reference.split())
    out = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        user_chunk = " ".join(user_answer.split()[i1:i2])
        ref_chunk = " ".join(reference.split()[j1:j2])
        if op == "equal":
            out.append(f'<span class="sg-eq">{user_chunk}</span>')
        elif op == "replace":
            out.append(f'<span class="sg-del">{user_chunk}</span> <span class="sg-ins">{ref_chunk}</span>')
        elif op == "delete":
            out.append(f'<span class="sg-del">{user_chunk}</span>')
        elif op == "insert":
            out.append(f'<span class="sg-ins">{ref_chunk}</span>')
    return " ".join(out)


def evaluate_answer(
    user_answer: str,
    reference: str,
    keywords: list[str],
    threshold: float,
    reference_embedding: list[float] | None = None,
) -> dict:
    """
    Run the full pipeline. Always returns a dict with `pass`, `keyword_check`,
    `similarity` (None if skipped), `threshold`, and `diff_html`.

    Keyword gate is strict: if any required keyword is missing, similarity is
    not even computed — we fail fast and tell the user which words to add.

    If `reference_embedding` is supplied (the calibration pass stores it on the
    note), we skip re-embedding the reference entirely.
    """
    user_answer = user_answer.strip()
    kw_result = check_keywords(user_answer, keywords)

    if not kw_result["ok"]:
        return {
            "pass": False,
            "reason": "missing_keywords",
            "keyword_check": kw_result,
            "similarity": None,
            "threshold": threshold,
            "diff_html": _diff_html(user_answer, reference),
        }

    try:
        ref_vec = reference_embedding if reference_embedding is not None else embed(reference)
        sim = cosine(embed(user_answer), ref_vec)
    except Exception as e:
        # Embedding failed (network, bad key, etc.) — surface it but don't crash review.
        return {
            "pass": False,
            "reason": "embedding_error",
            "error": str(e),
            "keyword_check": kw_result,
            "similarity": None,
            "threshold": threshold,
            "diff_html": _diff_html(user_answer, reference),
        }

    return {
        "pass": sim >= threshold,
        "reason": "ok" if sim >= threshold else "low_similarity",
        "keyword_check": kw_result,
        "similarity": sim,
        "threshold": threshold,
        "diff_html": _diff_html(user_answer, reference),
    }
