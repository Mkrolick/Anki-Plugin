"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def small_pdf_path() -> Path:
    return FIXTURES / "small.pdf"


class FakeLLM:
    """Inject canned JSON responses into pipeline modules that take an LLM client."""
    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, *, model: str, system: str, user: str, **_) -> dict[str, Any]:
        self.calls.append({"model": model, "system": system, "user": user})
        if not self._responses:
            raise RuntimeError("FakeLLM ran out of canned responses")
        return self._responses.pop(0)


class FakeEmbedder:
    """Inject deterministic vectors keyed by text."""
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self._vectors:
            return self._vectors[text]
        raise KeyError(f"FakeEmbedder has no vector for {text!r}")


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def fake_embedder():
    return FakeEmbedder
