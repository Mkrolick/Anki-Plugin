"""Merge stage — cluster topic strings via cosine on embeddings.

Avoids pulling in scipy / sklearn (Anki ships stdlib + our vendored deps).
A small agglomerative single-link clusterer over union-find suffices: for
~500 topics the O(n^2) pairwise scan is fine.
"""
from __future__ import annotations

import math
from typing import Protocol

from .types import Aspect, Topic


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (dot / (na * nb))


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def _centroid_nearest(tags: list[str], embeddings: dict[str, list[float]]) -> str:
    if len(tags) == 1:
        return tags[0]
    dim = len(embeddings[tags[0]])
    centroid = [0.0] * dim
    for t in tags:
        for k, v in enumerate(embeddings[t]):
            centroid[k] += v
    centroid = [x / len(tags) for x in centroid]
    return min(tags, key=lambda t: _cosine_distance(embeddings[t], centroid))


def merge_topics(
    aspects: list[Aspect],
    *,
    embedder: Embedder,
    distance_threshold: float = 0.25,
) -> list[Topic]:
    """Cluster aspect.topic strings and bundle each aspect under its cluster."""
    uniq: list[str] = []
    seen: set[str] = set()
    for a in aspects:
        if a.topic not in seen:
            seen.add(a.topic)
            uniq.append(a.topic)

    embeddings: dict[str, list[float]] = {}
    uncategorized: list[str] = []
    for tag in uniq:
        try:
            embeddings[tag] = embedder.embed(tag)
        except Exception:
            uncategorized.append(tag)

    embeddable = [t for t in uniq if t in embeddings]

    uf = _UnionFind(len(embeddable))
    for i in range(len(embeddable)):
        for j in range(i + 1, len(embeddable)):
            d = _cosine_distance(embeddings[embeddable[i]], embeddings[embeddable[j]])
            if d < distance_threshold:
                uf.union(i, j)

    clusters: dict[int, list[str]] = {}
    for i, tag in enumerate(embeddable):
        root = uf.find(i)
        clusters.setdefault(root, []).append(tag)

    topics: list[Topic] = []
    for tags in clusters.values():
        canonical = _centroid_nearest(tags, embeddings)
        topics.append(Topic(canonical=canonical, aliases=tags, aspects=[]))

    if uncategorized:
        topics.append(Topic(canonical="uncategorized", aliases=uncategorized, aspects=[]))

    tag_to_topic: dict[str, Topic] = {}
    for t in topics:
        for alias in t.aliases:
            tag_to_topic[alias] = t
    for a in aspects:
        if a.topic in tag_to_topic:
            tag_to_topic[a.topic].aspects.append(a)

    return topics
