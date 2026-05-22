"""Sliding-window iterator over pages.

step = size - overlap. The LAST chunk is anchored to the end of the book
so the tail page always appears in some chunk even when the book length
doesn't divide evenly.
"""
from __future__ import annotations

from typing import Iterator

from .types import Page


def chunk_iter(pages: list[Page], *, size: int, overlap: int) -> Iterator[list[Page]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")
    if not pages:
        return
    step = size - overlap
    n = len(pages)
    if n <= size:
        yield list(pages)
        return
    start = 0
    last_end = -1
    while True:
        end = min(start + size, n)
        chunk = pages[start:end]
        if chunk:
            yield chunk
        last_end = end
        if end >= n:
            break
        start += step
    if last_end < n:
        yield pages[max(0, n - size):n]
