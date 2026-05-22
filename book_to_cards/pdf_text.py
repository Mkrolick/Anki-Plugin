"""PDF → list[Page] via the vendored pdfminer.six.

We attach a 0-based page index to each chunk of extracted text and collapse
runs of whitespace so downstream LLM calls see clean prose without the
layout-driven line breaks pdfminer emits.
"""
from __future__ import annotations

import os
import re
import sys

_VENDOR = os.path.join(os.path.dirname(__file__), "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from pdfminer.high_level import extract_pages as _pdfminer_pages  # noqa: E402
from pdfminer.layout import LTTextContainer  # noqa: E402

from .pipeline.types import Page

# Pages with fewer than this many non-whitespace characters are dropped.
# Tuned to skip PDF artefacts (running headers, blank cover pages) without
# discarding genuine short pages like chapter title pages.
_USEFUL_PAGE_MIN_CHARS = 30

_WS_RUN = re.compile(r"\s+")


def _is_useful_page(text: str) -> bool:
    return sum(1 for c in text if not c.isspace()) >= _USEFUL_PAGE_MIN_CHARS


def _normalize(text: str) -> str:
    return _WS_RUN.sub(" ", text).strip()


def extract_pages(pdf_path: str) -> list[Page]:
    """Return Page(index, text) for every useful page in the PDF."""
    out: list[Page] = []
    for i, layout in enumerate(_pdfminer_pages(pdf_path)):
        parts: list[str] = []
        for element in layout:
            if isinstance(element, LTTextContainer):
                parts.append(element.get_text())
        text = _normalize("".join(parts))
        if _is_useful_page(text):
            out.append(Page(index=i, text=text))
    return out
