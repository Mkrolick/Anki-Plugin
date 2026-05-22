"""Shared dataclasses for the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Page:
    index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Aspect:
    text: str
    quote: str
    topic: str
    source_pages: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Topic:
    canonical: str
    aliases: list[str]
    aspects: list[Aspect] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "aliases": self.aliases,
            "aspects": [a.to_dict() for a in self.aspects],
        }


@dataclass
class Card:
    front: str
    back: str
    topic: str
    source_pages: list[int]
    source_quotes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
