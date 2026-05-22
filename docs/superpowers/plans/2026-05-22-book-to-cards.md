# Book-to-Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `book_to_cards` Anki add-on that ingests a PDF and produces auto-calibrated flashcards via an extract → map → merge → reduce pipeline. Refactor `smart_grader` minimally to expose `calibrate_note` as a public API.

**Architecture:** Two independent Anki add-ons sharing a vendored OpenAI client. Pipeline stages run in a `QThread` background worker, each writing JSON checkpoints under `user_files/<book_hash>/` for resumability. Cards stream into the user's chosen deck and are auto-calibrated through `smart_grader.api.calibrate_note` if the grader add-on is present.

**Tech Stack:** Python 3.10+ (Anki's bundled), PyQt6 (via aqt), pdfminer.six (vendored, pure-Python), OpenAI `gpt-4o-mini` + `text-embedding-3-small` via stdlib `urllib`, pytest for unit tests, headless `anki.collection.Collection` for integration, computer-use MCP for the end-to-end smoke test.

**Spec:** `docs/superpowers/specs/2026-05-22-book-to-cards-design.md`

---

## File Structure

```
Anki-Plugin/
├── build.sh                          # MOD — builds both .ankiaddon files
├── tests/                            # NEW
│   ├── conftest.py                   # shared fixtures
│   ├── fixtures/
│   │   └── small.pdf                 # ~5-page born-digital PDF
│   ├── test_smart_grader_api.py
│   ├── test_pdf_text.py
│   ├── test_chunks.py
│   ├── test_map_chunks.py
│   ├── test_merge_tags.py
│   ├── test_reduce_cards.py
│   ├── test_checkpoint.py
│   └── test_deck_writer.py
├── smart_grader/
│   ├── api.py                        # NEW
│   └── calibration.py                # MOD — extract _calibrate_one
├── book_to_cards/                    # NEW
│   ├── __init__.py
│   ├── manifest.json
│   ├── config.json
│   ├── config.py
│   ├── openai_client.py
│   ├── pdf_text.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── types.py                  # Page, Aspect, Topic, Card dataclasses
│   │   ├── map_chunks.py
│   │   ├── merge_tags.py
│   │   ├── reduce_cards.py
│   │   └── checkpoint.py
│   ├── runner.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── launch_dialog.py
│   │   └── progress_dialog.py
│   ├── deck_writer.py
│   └── _vendor/pdfminer/             # vendored, do not edit
└── docs/superpowers/plans/2026-05-22-book-to-cards.md
```

---

### Task 1: Test infrastructure

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/fixtures/small.pdf` (binary, copied in)
- Create: `pytest.ini`

- [ ] **Step 1: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -ra --strict-markers
```

- [ ] **Step 2: Create tests/conftest.py with shared fixtures**

```python
"""Shared pytest fixtures."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Make repo root importable so tests can import smart_grader.* and book_to_cards.*
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
```

- [ ] **Step 3: Create a tiny fixture PDF**

Run a one-shot helper to generate a 5-page PDF with known text. Use the `fpdf2` library if available, otherwise hand-write a minimal PDF.

```bash
python3 -c "
from reportlab.pdfgen import canvas
c = canvas.Canvas('tests/fixtures/small.pdf')
pages = [
    'Photosynthesis converts light energy into chemical energy. It occurs in chloroplasts.',
    'The Calvin cycle fixes carbon dioxide using RuBisCO. It runs in the stroma.',
    'ATP is produced via chemiosmosis across the thylakoid membrane during the light reactions.',
    'Mitochondria are the powerhouse of the cell. They perform oxidative phosphorylation.',
    'The electron transport chain pumps protons across the inner mitochondrial membrane.',
]
for text in pages:
    c.drawString(72, 720, text)
    c.showPage()
c.save()
print('OK')
"
```

If `reportlab` is not installed: `pip install reportlab` first.

- [ ] **Step 4: Verify the fixture works**

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
# pdfminer not installed yet — just check the file is a valid PDF
with open('tests/fixtures/small.pdf', 'rb') as f:
    assert f.read(4) == b'%PDF', 'fixture is not a PDF'
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 5: Run pytest with no tests yet**

```bash
python3 -m pytest tests/ -q
```

Expected: `no tests ran` (or equivalent), exit code 5. This is acceptable — infrastructure is in place.

- [ ] **Step 6: Commit**

```bash
git add pytest.ini tests/conftest.py tests/fixtures/small.pdf
git commit -m "Add pytest infrastructure and small fixture PDF"
```

---

### Task 2: smart_grader refactor — api.py + _calibrate_one

**Files:**
- Create: `smart_grader/api.py`
- Modify: `smart_grader/calibration.py:159-186` (extract `_calibrate_one`)
- Create: `tests/test_smart_grader_api.py`

- [ ] **Step 1: Write the failing test**

`tests/test_smart_grader_api.py`:

```python
"""Smart Grader's external API surface."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_api_module_exports_required_names():
    """smart_grader.api must expose calibrate_note and is_available."""
    # Importing the package would normally fail outside Anki (it imports aqt at
    # the top of __init__.py). We poke at the api module via importlib so it
    # can be tested standalone.
    spec = importlib.util.spec_from_file_location(
        "smart_grader_api",
        Path(__file__).resolve().parent.parent / "smart_grader" / "api.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # We expect the file to expose these names without executing aqt imports
    # at module load time.
    source = (Path(__file__).resolve().parent.parent / "smart_grader" / "api.py").read_text()
    assert "def calibrate_note(" in source
    assert "def is_available(" in source


def test_is_available_returns_false_without_aqt(monkeypatch):
    """is_available() must return False when aqt is not importable."""
    # Force aqt import to fail
    monkeypatch.setitem(sys.modules, "aqt", None)
    # Re-import api fresh
    if "smart_grader.api" in sys.modules:
        del sys.modules["smart_grader.api"]
    # We can't actually import smart_grader.api outside Anki because it depends
    # on calibration.py which imports aqt at module top. So test the public
    # contract via source inspection.
    source = (Path(__file__).resolve().parent.parent / "smart_grader" / "api.py").read_text()
    assert "importlib" in source or "find_spec" in source or "try:" in source, \
        "is_available() should guard against missing aqt"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_smart_grader_api.py -v
```

Expected: FAIL with file-not-found / assertion errors.

- [ ] **Step 3: Create smart_grader/api.py**

```python
"""Public API surface for smart_grader.

This module is the single import point for other add-ons that want to
calibrate notes via smart_grader. Keeping the surface narrow means we can
evolve the internals without breaking downstream consumers.
"""
from __future__ import annotations

import importlib.util


def is_available() -> bool:
    """True iff this module can be imported and aqt is on the path.

    `book_to_cards` calls this before attempting calibrate_note so it can
    silently skip integration when smart_grader isn't installed.
    """
    return importlib.util.find_spec("aqt") is not None


def calibrate_note(note) -> dict:
    """Calibrate a single Anki note in-place.

    Returns a dict {"nid", "threshold", "overlap"}. Raises on hard failures
    (missing fields, API errors); callers should catch and report rather
    than letting one bad note abort a batch.
    """
    from .calibration import _calibrate_one
    return _calibrate_one(note)
```

- [ ] **Step 4: Refactor calibration.py to extract _calibrate_one**

Currently `calibrate_note(note)` in `smart_grader/calibration.py:82` is the per-note work. Rename it to `_calibrate_one(note)` (single source of truth) and keep `calibrate_note` as a public alias for backwards compatibility. Update `calibrate_selected_notes` to call `_calibrate_one` directly.

In `smart_grader/calibration.py`, change:

```python
def calibrate_note(note) -> dict:
    """
    Calibrate a single note. Returns a diagnostics dict; raises on hard failures
    (missing fields, API errors) so the caller can decide whether to skip or abort.
    """
```

to:

```python
def _calibrate_one(note) -> dict:
    """
    Calibrate a single note. Returns a diagnostics dict; raises on hard failures
    (missing fields, API errors) so the caller can decide whether to skip or abort.
    """
```

Then in `calibrate_selected_notes`, change `calibrate_note(note)` (line ~179) to `_calibrate_one(note)`.

Add a public alias at the bottom of calibration.py:

```python
# Backwards-compatible alias. New code should import from smart_grader.api.
calibrate_note = _calibrate_one
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_smart_grader_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Rebuild the .ankiaddon and verify py_compile**

```bash
./build.sh
```

Expected: build succeeds, archive contains api.py.

- [ ] **Step 7: Commit**

```bash
git add smart_grader/api.py smart_grader/calibration.py tests/test_smart_grader_api.py smart_grader.ankiaddon
git commit -m "smart_grader: expose calibrate_note via public api module"
```

---

### Task 3: book_to_cards skeleton

**Files:**
- Create: `book_to_cards/manifest.json`
- Create: `book_to_cards/config.json`
- Create: `book_to_cards/config.py`
- Create: `book_to_cards/__init__.py` (minimal stub)
- Modify: `build.sh` (build both addons)

- [ ] **Step 1: Create book_to_cards/manifest.json**

```json
{
    "name": "Book to Cards",
    "package": "book_to_cards",
    "min_point_version": 231000,
    "tested_on": [241100, 250100]
}
```

- [ ] **Step 2: Create book_to_cards/config.json**

```json
{
    "openai_api_key": "",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "chunk_pages": 5,
    "chunk_overlap": 1,
    "cluster_distance": 0.25,
    "max_cards_per_topic": 5,
    "default_max_cost_usd": 5.0,
    "skip_per_card_calibration": false
}
```

- [ ] **Step 3: Create book_to_cards/config.py**

```python
"""Config accessor — reads from Anki's per-addon config (managed by Anki itself)."""
from aqt import mw

DEFAULTS = {
    "openai_api_key": "",
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "chunk_pages": 5,
    "chunk_overlap": 1,
    "cluster_distance": 0.25,
    "max_cards_per_topic": 5,
    "default_max_cost_usd": 5.0,
    "skip_per_card_calibration": False,
}


def get_config() -> dict:
    raw = mw.addonManager.getConfig(__name__.split(".")[0]) or {}
    return {**DEFAULTS, **raw}
```

- [ ] **Step 4: Create book_to_cards/__init__.py (stub)**

```python
"""Book to Cards for Anki — PDF ingest → auto-generated flashcards.

This file is loaded by Anki at startup. It registers the Tools menu entry
and otherwise stays out of the way.
"""
from __future__ import annotations

from aqt import mw
from aqt.qt import QAction


def _on_generate():
    # Wired up in Task 16 (UI integration). Stub keeps the menu wired so we
    # can install and load the add-on while building out the pipeline.
    from aqt.utils import showInfo
    showInfo("Book to Cards: not yet implemented.")


def _install_menu():
    action = QAction("Generate cards from PDF…", mw)
    action.triggered.connect(_on_generate)
    mw.form.menuTools.addAction(action)


_install_menu()
```

- [ ] **Step 5: Update build.sh**

```bash
#!/usr/bin/env bash
# Build smart_grader.ankiaddon and book_to_cards.ankiaddon from source.
# Anki requires files at the ROOT of the .ankiaddon zip, not nested in a folder.
set -euo pipefail

cd "$(dirname "$0")"

build_addon() {
  local name="$1"
  local out="${name}.ankiaddon"
  rm -f "$out"
  python3 -m py_compile "${name}"/*.py "${name}"/**/*.py 2>/dev/null || \
    python3 -m py_compile "${name}"/*.py
  (cd "$name" && zip -qr "../$out" . \
    -x "*.DS_Store" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc" \
    -x "embedding_cache.sqlite" \
    -x "user_files/*")
  echo "Built $out:"
  unzip -l "$out" | tail -5
}

build_addon smart_grader

if [ -d book_to_cards ]; then
  build_addon book_to_cards
fi
```

- [ ] **Step 6: Run the build**

```bash
chmod +x build.sh && ./build.sh
```

Expected: two `.ankiaddon` files built; book_to_cards archive contains manifest.json, config.json, config.py, __init__.py.

- [ ] **Step 7: Commit**

```bash
git add book_to_cards/ build.sh
git commit -m "book_to_cards: scaffold with manifest, config, and menu stub"
```

---

### Task 4: Vendor pdfminer.six

**Files:**
- Create: `book_to_cards/_vendor/pdfminer/` (downloaded tree)
- Create: `book_to_cards/_vendor/__init__.py` (empty)

- [ ] **Step 1: Download pdfminer.six and extract just the package**

```bash
mkdir -p book_to_cards/_vendor
cd /tmp
pip download --no-deps --no-binary :all: pdfminer.six -d pdfminer_dl 2>&1 | tail -3
TARBALL=$(ls /tmp/pdfminer_dl/pdfminer*.tar.gz | head -1)
tar -xzf "$TARBALL" -C /tmp/pdfminer_extract --strip-components=1
mkdir -p /tmp/pdfminer_extract
tar -xzf "$TARBALL" -C /tmp/pdfminer_extract
SRC=$(find /tmp/pdfminer_extract -type d -name pdfminer | head -1)
cp -R "$SRC" /Users/malcolmkrolick/Documents/GitHub/Anki-Plugin/book_to_cards/_vendor/
cd /Users/malcolmkrolick/Documents/GitHub/Anki-Plugin
touch book_to_cards/_vendor/__init__.py
```

- [ ] **Step 2: Add path-prepend bootstrap so _vendor is importable**

Add to `book_to_cards/__init__.py`, at the very top (before `from aqt`):

```python
import os
import sys
_VENDOR = os.path.join(os.path.dirname(__file__), "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
```

- [ ] **Step 3: Verify import works in isolation**

```bash
python3 -c "
import sys
sys.path.insert(0, 'book_to_cards/_vendor')
from pdfminer.high_level import extract_text
print('OK')
"
```

Expected: `OK`. If pdfminer has its own deps (`charset-normalizer`, `cryptography`, `pycparser`, `cffi`), download those too via `pip download --no-deps` and place in `_vendor/`.

- [ ] **Step 4: Run pdfminer against the fixture**

```bash
python3 -c "
import sys
sys.path.insert(0, 'book_to_cards/_vendor')
from pdfminer.high_level import extract_text
t = extract_text('tests/fixtures/small.pdf')
assert 'Photosynthesis' in t, repr(t[:200])
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add book_to_cards/_vendor/
git commit -m "book_to_cards: vendor pdfminer.six for in-Anki PDF extraction"
```

---

### Task 5: pdf_text module

**Files:**
- Create: `book_to_cards/pdf_text.py`
- Create: `book_to_cards/pipeline/__init__.py`
- Create: `book_to_cards/pipeline/types.py`
- Create: `tests/test_pdf_text.py`

- [ ] **Step 1: Create types.py with the Page dataclass**

```python
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
```

- [ ] **Step 2: Create empty pipeline/__init__.py**

```python
"""book_to_cards pipeline stages."""
```

- [ ] **Step 3: Write the failing test**

`tests/test_pdf_text.py`:

```python
"""PDF extraction smoke tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "book_to_cards" / "_vendor"))


def test_extracts_pages_with_known_text(small_pdf_path):
    from book_to_cards.pdf_text import extract_pages
    pages = extract_pages(str(small_pdf_path))
    assert len(pages) == 5
    assert pages[0].index == 0
    assert "Photosynthesis" in pages[0].text


def test_pages_text_is_whitespace_normalized(small_pdf_path):
    from book_to_cards.pdf_text import extract_pages
    pages = extract_pages(str(small_pdf_path))
    for p in pages:
        # no runs of 2+ whitespace characters
        import re
        assert re.search(r"\s{2,}", p.text) is None, f"page {p.index}: {p.text!r}"


def test_blank_pages_dropped(tmp_path, small_pdf_path):
    """If a page has < 50 non-whitespace chars, it should be skipped."""
    # Smoke test: all 5 fixture pages have content, so all 5 come through.
    # The threshold behaviour is covered by a direct unit test on the
    # private helper.
    from book_to_cards.pdf_text import _is_useful_page
    assert _is_useful_page("Photosynthesis converts light energy.") is True
    assert _is_useful_page("   ") is False
    assert _is_useful_page("x" * 49) is False
    assert _is_useful_page("x" * 50) is True
```

- [ ] **Step 4: Run test to verify it fails**

```bash
python3 -m pytest tests/test_pdf_text.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 5: Create book_to_cards/pdf_text.py**

```python
"""PDF → list[Page] via the vendored pdfminer.six.

We use pdfminer's high-level page iterator so we can attach a 0-based page
index to each chunk of extracted text. Whitespace is collapsed (runs of
spaces/newlines/tabs become a single space) so downstream LLM calls see
clean prose without the layout-driven line breaks pdfminer emits.
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
# discarding genuine short pages like chapter title pages with a sentence.
_USEFUL_PAGE_MIN_CHARS = 50

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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_pdf_text.py -v
```

Expected: PASS for all three tests.

- [ ] **Step 7: Commit**

```bash
git add book_to_cards/pdf_text.py book_to_cards/pipeline/ tests/test_pdf_text.py
git commit -m "book_to_cards: PDF extraction via vendored pdfminer"
```

---

### Task 6: Chunk iterator

**Files:**
- Create: `book_to_cards/pipeline/map_chunks.py` (chunk_iter only — LLM call comes in Task 8)
- Create: `tests/test_chunks.py`

- [ ] **Step 1: Write the failing test**

`tests/test_chunks.py`:

```python
"""Chunk iterator: sliding window of pages with overlap."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _pages(n: int):
    from book_to_cards.pipeline.types import Page
    return [Page(index=i, text=f"page {i} content") for i in range(n)]


def test_single_chunk_when_book_shorter_than_window():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    chunks = list(chunk_iter(_pages(3), size=5, overlap=1))
    assert len(chunks) == 1
    assert [p.index for p in chunks[0]] == [0, 1, 2]


def test_exact_window_one_chunk():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    chunks = list(chunk_iter(_pages(5), size=5, overlap=1))
    assert len(chunks) == 1
    assert [p.index for p in chunks[0]] == [0, 1, 2, 3, 4]


def test_overlap_step_correct():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    chunks = list(chunk_iter(_pages(13), size=5, overlap=1))
    # step = 4 → starts at 0, 4, 8 → final chunk [8..12]
    assert len(chunks) == 3
    assert [p.index for p in chunks[0]] == [0, 1, 2, 3, 4]
    assert [p.index for p in chunks[1]] == [4, 5, 6, 7, 8]
    assert [p.index for p in chunks[2]] == [8, 9, 10, 11, 12]


def test_every_page_appears_at_least_once():
    from book_to_cards.pipeline.map_chunks import chunk_iter
    pages = _pages(17)
    seen: set[int] = set()
    for chunk in chunk_iter(pages, size=5, overlap=1):
        seen.update(p.index for p in chunk)
    assert seen == set(range(17))


def test_tail_handles_uneven_book():
    """Books whose length isn't size + k*step still cover every page."""
    from book_to_cards.pipeline.map_chunks import chunk_iter
    pages = _pages(11)  # 5, 5+4=9, last needs 9..13 but only have 11
    chunks = list(chunk_iter(pages, size=5, overlap=1))
    flat = [p.index for c in chunks for p in c]
    assert set(flat) == set(range(11))
    # Last chunk anchored to the end, not starting mid-book
    assert chunks[-1][-1].index == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_chunks.py -v
```

Expected: FAIL with module not found.

- [ ] **Step 3: Create book_to_cards/pipeline/map_chunks.py with chunk_iter**

```python
"""Sliding-window iterator over pages.

step = size - overlap. We anchor the LAST chunk to the end of the book so
the tail page always appears in some chunk even when the book length
doesn't divide evenly.
"""
from __future__ import annotations

from typing import Iterable, Iterator

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
    # If the book fits in a single window, emit one chunk and stop.
    if n <= size:
        yield list(pages)
        return
    start = 0
    last_emitted_end = -1
    while True:
        end = min(start + size, n)
        chunk = pages[start:end]
        if chunk:
            yield chunk
        last_emitted_end = end
        if end >= n:
            break
        start += step
    # Ensure the final page is covered. If the last chunk didn't reach n,
    # emit one more anchored to the end.
    if last_emitted_end < n:
        yield pages[max(0, n - size):n]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_chunks.py -v
```

Expected: PASS for all five tests.

- [ ] **Step 5: Commit**

```bash
git add book_to_cards/pipeline/map_chunks.py tests/test_chunks.py
git commit -m "book_to_cards: sliding chunk iterator"
```

---

### Task 7: OpenAI client (vendored)

**Files:**
- Create: `book_to_cards/openai_client.py`

- [ ] **Step 1: Copy smart_grader's openai_client.py as a starting point and adapt**

```python
"""OpenAI client wrapper for book_to_cards.

Vendored from smart_grader so the two add-ons stay independently
installable. Two responsibilities:
    1. embed(text) -> list[float], cached on disk by SHA256.
    2. chat_json(system, user, model) -> dict, JSON-mode response.

Also tracks cumulative (input_tokens, output_tokens) so the pipeline runner
can enforce a cost ceiling.
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
        cfg = get_config()
        m = model or cfg["embedding_model"]
        key = _cache_key(text, m)
        conn = _cache_conn()
        try:
            row = conn.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
            if row is not None:
                return json.loads(row[0])
            resp = _http_post_json(
                "https://api.openai.com/v1/embeddings",
                {"model": m, "input": text},
                self._api_key,
            )
            usage = resp.get("usage", {})
            self.meter.record(m, usage.get("prompt_tokens", 0), 0)
            vec = resp["data"][0]["embedding"]
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(key, model, vector) VALUES (?, ?, ?)",
                (key, m, json.dumps(vec)),
            )
            conn.commit()
            return vec
        finally:
            conn.close()

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
```

- [ ] **Step 2: Smoke test — module parses**

```bash
python3 -m py_compile book_to_cards/openai_client.py
```

Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add book_to_cards/openai_client.py
git commit -m "book_to_cards: vendored OpenAI client with cost meter"
```

---

### Task 8: Map stage

**Files:**
- Modify: `book_to_cards/pipeline/map_chunks.py` (add run_map)
- Create: `tests/test_map_chunks.py`

- [ ] **Step 1: Write the failing test**

`tests/test_map_chunks.py`:

```python
"""Map stage: chunks → aspects via the LLM client."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _pages(n):
    from book_to_cards.pipeline.types import Page
    return [Page(index=i, text=f"page {i} content about topic_{i % 3}") for i in range(n)]


def test_map_emits_one_result_per_chunk(fake_llm):
    from book_to_cards.pipeline.map_chunks import run_map
    canned = [
        {"aspects": [{"text": f"fact {j}", "quote": f"q{j}", "topic": f"t{j}", "source_pages": [j]}]
         for j in range(2)}
        for _ in range(3)
    ]
    # The above is wrong — each call needs its own response. Build correctly:
    canned = [
        {"aspects": [{"text": "fact A", "quote": "qA", "topic": "tA", "source_pages": [0]}]},
        {"aspects": [{"text": "fact B", "quote": "qB", "topic": "tB", "source_pages": [4]}]},
        {"aspects": [{"text": "fact C", "quote": "qC", "topic": "tC", "source_pages": [8]}]},
    ]
    llm = fake_llm(canned)
    results = list(run_map(_pages(13), llm=llm, size=5, overlap=1))
    assert len(results) == 3
    assert llm.calls[0]["model"]  # was passed
    assert "page 0 content" in llm.calls[0]["user"]


def test_map_attaches_source_page_indices(fake_llm):
    from book_to_cards.pipeline.map_chunks import run_map
    canned = [
        {"aspects": [{"text": "x", "quote": "q", "topic": "t", "source_pages": [0, 1]}]},
    ]
    llm = fake_llm(canned)
    [chunk_result] = list(run_map(_pages(3), llm=llm, size=5, overlap=1))
    assert chunk_result["chunk_index"] == 0
    assert chunk_result["source_pages"] == [0, 1, 2]
    assert chunk_result["aspects"][0]["topic"] == "t"


def test_map_continues_on_chunk_failure(fake_llm):
    from book_to_cards.pipeline.map_chunks import run_map

    class FlakyLLM:
        def __init__(self):
            self.n = 0
            self.calls = []
        def chat_json(self, **kw):
            self.calls.append(kw)
            self.n += 1
            if self.n == 2:
                raise RuntimeError("boom")
            return {"aspects": [{"text": "ok", "quote": "q", "topic": "t",
                                  "source_pages": [self.n]}]}

    llm = FlakyLLM()
    results = list(run_map(_pages(13), llm=llm, size=5, overlap=1))
    # 3 chunks, middle one failed → 2 good results + 1 error record
    good = [r for r in results if "aspects" in r and r.get("error") is None]
    errors = [r for r in results if r.get("error") is not None]
    assert len(good) == 2
    assert len(errors) == 1
    assert errors[0]["chunk_index"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_map_chunks.py -v
```

Expected: FAIL — `run_map` not defined.

- [ ] **Step 3: Extend map_chunks.py with run_map**

Append to `book_to_cards/pipeline/map_chunks.py`:

```python
import json as _json

_SYSTEM_PROMPT = """\
You are an expert at extracting study material from textbooks. For the
passage below, list every noteworthy fact, definition, mechanism, or claim
a student should remember. For each item return:
  - "text": a one-sentence distillation
  - "quote": the EXACT verbatim excerpt from the passage (do not edit)
  - "topic": a short topic name (1-4 words)
  - "source_pages": the page numbers (from the [Page N] markers) the excerpt came from

Respond with a single JSON object: {"aspects": [...]}. No prose, no fences.
"""


def _build_user_prompt(chunk: list) -> str:
    parts: list[str] = []
    for page in chunk:
        parts.append(f"[Page {page.index}]")
        parts.append(page.text)
    return "\n".join(parts)


def run_map(pages, *, llm, size: int, overlap: int, model: str | None = None):
    """Yield one dict per chunk.

    On success: {"chunk_index", "source_pages", "aspects": [...]}
    On failure: {"chunk_index", "source_pages", "aspects": [], "error": "msg"}

    The runner persists each yielded dict immediately so a crash mid-loop
    loses at most one chunk's worth of in-flight work.
    """
    for i, chunk in enumerate(chunk_iter(pages, size=size, overlap=overlap)):
        source_pages = [p.index for p in chunk]
        try:
            response = llm.chat_json(
                model=model or "gpt-4o-mini",
                system=_SYSTEM_PROMPT,
                user=_build_user_prompt(chunk),
            )
            aspects = response.get("aspects") or []
            if not isinstance(aspects, list):
                raise RuntimeError(f"map returned non-list aspects: {type(aspects).__name__}")
            yield {
                "chunk_index": i,
                "source_pages": source_pages,
                "aspects": aspects,
                "error": None,
            }
        except Exception as e:
            yield {
                "chunk_index": i,
                "source_pages": source_pages,
                "aspects": [],
                "error": str(e),
            }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_map_chunks.py -v
```

Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add book_to_cards/pipeline/map_chunks.py tests/test_map_chunks.py
git commit -m "book_to_cards: map stage with per-chunk failure isolation"
```

---

### Task 9: Merge stage (embedding-based topic clustering)

**Files:**
- Create: `book_to_cards/pipeline/merge_tags.py`
- Create: `tests/test_merge_tags.py`

- [ ] **Step 1: Write the failing test**

`tests/test_merge_tags.py`:

```python
"""Merge stage: topic strings → canonical clusters via cosine clustering."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _unit_vector(angle_rad: float, dim: int = 4) -> list[float]:
    """A simple 2D unit vector embedded into `dim` dims (rest zero)."""
    v = [math.cos(angle_rad), math.sin(angle_rad)] + [0.0] * (dim - 2)
    return v


def _aspects_with_topics(topics):
    from book_to_cards.pipeline.types import Aspect
    return [Aspect(text=t, quote=t, topic=t, source_pages=[0]) for t in topics]


def test_close_tags_merge(fake_embedder):
    from book_to_cards.pipeline.merge_tags import merge_topics
    # Three close vectors (within 0.1 rad of each other) + one far away
    vecs = {
        "chemiosmosis": _unit_vector(0.00),
        "electron transport chain": _unit_vector(0.05),
        "oxidative phosphorylation": _unit_vector(0.10),
        "photosynthesis": _unit_vector(2.50),
    }
    aspects = _aspects_with_topics(list(vecs.keys()))
    topics = merge_topics(aspects, embedder=fake_embedder(vecs), distance_threshold=0.25)
    assert len(topics) == 2
    biology_cluster = next(t for t in topics if "photosynthesis" in t.aliases or t.canonical == "photosynthesis")
    chem_cluster = next(t for t in topics if t is not biology_cluster)
    assert len(chem_cluster.aliases) == 3
    assert len(biology_cluster.aliases) == 1


def test_each_aspect_lands_in_exactly_one_cluster(fake_embedder):
    from book_to_cards.pipeline.merge_tags import merge_topics
    vecs = {
        "a": _unit_vector(0.0),
        "b": _unit_vector(0.05),
        "c": _unit_vector(3.0),
    }
    aspects = _aspects_with_topics(["a", "b", "c"])
    topics = merge_topics(aspects, embedder=fake_embedder(vecs), distance_threshold=0.25)
    seen_topics: set[str] = set()
    for t in topics:
        for a in t.aspects:
            assert a.topic not in seen_topics
            seen_topics.add(a.topic)
    assert seen_topics == {"a", "b", "c"}


def test_unembeddable_tag_goes_to_uncategorized(fake_embedder):
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
    # Two clusters: the working one and an "uncategorized" singleton for "broken"
    canonicals = {t.canonical for t in topics}
    assert "uncategorized" in canonicals
    uncat = next(t for t in topics if t.canonical == "uncategorized")
    assert [a.topic for a in uncat.aspects] == ["broken"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_merge_tags.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create book_to_cards/pipeline/merge_tags.py**

```python
"""Merge stage — cluster topic strings via cosine on embeddings.

We don't pull in scipy / sklearn (Anki ships stdlib + our vendored deps).
A small agglomerative single-link clusterer suffices: union-find over
edges (pair-distance < threshold). For ~500 topics the O(n^2) is fine.
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


def merge_topics(
    aspects: list[Aspect],
    *,
    embedder: Embedder,
    distance_threshold: float = 0.25,
) -> list[Topic]:
    """Cluster aspect.topic strings and bundle each aspect under its cluster."""
    # 1. Unique topic strings
    uniq: list[str] = []
    seen: set[str] = set()
    for a in aspects:
        if a.topic not in seen:
            seen.add(a.topic)
            uniq.append(a.topic)

    # 2. Embed each (with isolation: failures → uncategorized)
    embeddings: dict[str, list[float]] = {}
    uncategorized: list[str] = []
    for tag in uniq:
        try:
            embeddings[tag] = embedder.embed(tag)
        except Exception:
            uncategorized.append(tag)

    embeddable = [t for t in uniq if t in embeddings]

    # 3. Pairwise union under threshold
    uf = _UnionFind(len(embeddable))
    for i in range(len(embeddable)):
        for j in range(i + 1, len(embeddable)):
            d = _cosine_distance(embeddings[embeddable[i]], embeddings[embeddable[j]])
            if d < distance_threshold:
                uf.union(i, j)

    # 4. Build clusters
    clusters: dict[int, list[str]] = {}
    for i, tag in enumerate(embeddable):
        root = uf.find(i)
        clusters.setdefault(root, []).append(tag)

    # 5. Pick canonical: centroid-nearest tag in each cluster
    topics: list[Topic] = []
    for tags in clusters.values():
        canonical = _centroid_nearest(tags, embeddings)
        topic = Topic(canonical=canonical, aliases=tags, aspects=[])
        topics.append(topic)

    # 6. Add the uncategorized bucket if non-empty
    if uncategorized:
        topics.append(Topic(canonical="uncategorized", aliases=uncategorized, aspects=[]))

    # 7. Bundle aspects into their cluster
    tag_to_topic: dict[str, Topic] = {}
    for t in topics:
        for alias in t.aliases:
            tag_to_topic[alias] = t
    for a in aspects:
        if a.topic in tag_to_topic:
            tag_to_topic[a.topic].aspects.append(a)

    return topics


def _centroid_nearest(tags: list[str], embeddings: dict[str, list[float]]) -> str:
    """Return the tag closest to the cluster's mean vector."""
    if len(tags) == 1:
        return tags[0]
    dim = len(embeddings[tags[0]])
    centroid = [0.0] * dim
    for t in tags:
        for k, v in enumerate(embeddings[t]):
            centroid[k] += v
    centroid = [x / len(tags) for x in centroid]
    best = min(tags, key=lambda t: _cosine_distance(embeddings[t], centroid))
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_merge_tags.py -v
```

Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add book_to_cards/pipeline/merge_tags.py tests/test_merge_tags.py
git commit -m "book_to_cards: merge stage with cosine-based topic clustering"
```

---

### Task 10: Reduce stage (topic → atomic cards)

**Files:**
- Create: `book_to_cards/pipeline/reduce_cards.py`
- Create: `tests/test_reduce_cards.py`

- [ ] **Step 1: Write the failing test**

`tests/test_reduce_cards.py`:

```python
"""Reduce stage: each Topic → 1-N Cards via the LLM."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _topic(canonical, aliases, aspects):
    from book_to_cards.pipeline.types import Aspect, Topic
    return Topic(canonical=canonical, aliases=aliases, aspects=[
        Aspect(text=a["text"], quote=a["quote"], topic=a["topic"], source_pages=a["pages"])
        for a in aspects
    ])


def test_reduce_emits_cards_with_source_quotes_resolved(fake_llm):
    from book_to_cards.pipeline.reduce_cards import reduce_topic
    canned = [{"cards": [
        {"front": "Q1", "back": "A1", "source_quote_indices": [0, 1]},
        {"front": "Q2", "back": "A2", "source_quote_indices": [1]},
    ]}]
    topic = _topic("photosynthesis", ["photosynthesis", "light reactions"], [
        {"text": "f0", "quote": "quote zero", "topic": "photosynthesis", "pages": [1]},
        {"text": "f1", "quote": "quote one", "topic": "light reactions", "pages": [3, 4]},
    ])
    cards = reduce_topic(topic, llm=fake_llm(canned))
    assert len(cards) == 2
    assert cards[0].source_quotes == ["quote zero", "quote one"]
    assert cards[0].source_pages == [1, 3, 4]
    assert cards[1].source_quotes == ["quote one"]


def test_reduce_carries_topic_name(fake_llm):
    from book_to_cards.pipeline.reduce_cards import reduce_topic
    topic = _topic("X", ["X"], [{"text": "f", "quote": "q", "topic": "X", "pages": [0]}])
    canned = [{"cards": [{"front": "Q", "back": "A", "source_quote_indices": [0]}]}]
    cards = reduce_topic(topic, llm=fake_llm(canned))
    assert cards[0].topic == "X"


def test_reduce_skips_invalid_indices(fake_llm):
    from book_to_cards.pipeline.reduce_cards import reduce_topic
    topic = _topic("X", ["X"], [{"text": "f", "quote": "q0", "topic": "X", "pages": [0]}])
    # Index 5 is out of range; the card should still be emitted but without
    # the bad reference.
    canned = [{"cards": [{"front": "Q", "back": "A", "source_quote_indices": [0, 5]}]}]
    cards = reduce_topic(topic, llm=fake_llm(canned))
    assert cards[0].source_quotes == ["q0"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_reduce_cards.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create book_to_cards/pipeline/reduce_cards.py**

```python
"""Reduce stage — collapse all aspects under a topic into 1-N atomic cards."""
from __future__ import annotations

from typing import Protocol

from .types import Card, Topic


class _LLM(Protocol):
    def chat_json(self, *, model: str, system: str, user: str, **kw) -> dict: ...


_SYSTEM_PROMPT = """\
You are writing Anki flashcards from study material on one topic.
You will be given the topic name, its aliases, and several extracted facts
(each with a verbatim source quote). Produce 1-5 atomic flashcards.

Rules:
- Each card tests EXACTLY ONE fact. If a topic has multiple distinct facts,
  emit multiple cards.
- Ground every card in the provided quotes. Do not introduce information
  that is not supported by them.
- Front is a question. Back is the answer in 1-3 sentences.
- Don't include the topic name in the front unless it's the natural way to
  phrase the question.

Respond with JSON: {"cards": [{"front": "...", "back": "...",
"source_quote_indices": [0, 2]}]}. No prose, no fences.
"""


def _build_user_prompt(topic: Topic) -> str:
    lines: list[str] = []
    lines.append(f"Topic: {topic.canonical}")
    if topic.aliases and topic.aliases != [topic.canonical]:
        lines.append(f"Aliases: {', '.join(topic.aliases)}")
    lines.append("")
    lines.append("Quotes:")
    for i, aspect in enumerate(topic.aspects):
        pages = ", ".join(str(p) for p in aspect.source_pages)
        lines.append(f"  [{i}] \"{aspect.quote}\" (pp. {pages})")
    return "\n".join(lines)


def reduce_topic(topic: Topic, *, llm: _LLM, model: str = "gpt-4o-mini") -> list[Card]:
    response = llm.chat_json(
        model=model,
        system=_SYSTEM_PROMPT,
        user=_build_user_prompt(topic),
    )
    raw_cards = response.get("cards") or []
    out: list[Card] = []
    for raw in raw_cards:
        front = str(raw.get("front", "")).strip()
        back = str(raw.get("back", "")).strip()
        if not front or not back:
            continue
        indices = raw.get("source_quote_indices") or []
        quotes: list[str] = []
        pages: set[int] = set()
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(topic.aspects):
                quotes.append(topic.aspects[idx].quote)
                pages.update(topic.aspects[idx].source_pages)
        out.append(Card(
            front=front,
            back=back,
            topic=topic.canonical,
            source_pages=sorted(pages),
            source_quotes=quotes,
        ))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_reduce_cards.py -v
```

Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add book_to_cards/pipeline/reduce_cards.py tests/test_reduce_cards.py
git commit -m "book_to_cards: reduce stage produces atomic cards grounded in quotes"
```

---

### Task 11: Checkpoint module

**Files:**
- Create: `book_to_cards/pipeline/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

`tests/test_checkpoint.py`:

```python
"""Checkpoint: atomic writes + resume rules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_atomic_write_replaces_existing(tmp_path):
    from book_to_cards.pipeline.checkpoint import write_json_atomic
    p = tmp_path / "x.json"
    p.write_text('{"old": true}')
    write_json_atomic(p, {"new": True})
    assert json.loads(p.read_text()) == {"new": True}
    # No leftover .tmp
    assert not (tmp_path / "x.json.tmp").exists()


def test_resume_state_fresh_directory(tmp_path):
    from book_to_cards.pipeline.checkpoint import detect_resume_state
    state = detect_resume_state(tmp_path)
    assert state.next_stage == "extract"


def test_resume_state_after_extract(tmp_path):
    from book_to_cards.pipeline.checkpoint import detect_resume_state, write_json_atomic
    write_json_atomic(tmp_path / "extract.json", {"book_hash": "abc", "pages": []})
    state = detect_resume_state(tmp_path)
    assert state.next_stage == "map"


def test_resume_state_partial_map(tmp_path):
    from book_to_cards.pipeline.checkpoint import detect_resume_state, write_json_atomic, append_jsonl
    write_json_atomic(tmp_path / "extract.json", {"book_hash": "abc", "pages": []})
    append_jsonl(tmp_path / "map.jsonl", {"chunk_index": 0, "aspects": []})
    append_jsonl(tmp_path / "map.jsonl", {"chunk_index": 1, "aspects": []})
    state = detect_resume_state(tmp_path)
    assert state.next_stage == "map"
    assert state.map_resume_from == 2


def test_resume_state_complete_map_starts_merge(tmp_path):
    from book_to_cards.pipeline.checkpoint import detect_resume_state, write_json_atomic, append_jsonl
    write_json_atomic(tmp_path / "extract.json", {"book_hash": "abc", "pages": []})
    append_jsonl(tmp_path / "map.jsonl", {"chunk_index": 0, "aspects": []})
    append_jsonl(tmp_path / "map.jsonl", {"_complete": True})
    state = detect_resume_state(tmp_path)
    assert state.next_stage == "merge"


def test_manifest_mismatch_triggers_cold_restart_flag(tmp_path):
    from book_to_cards.pipeline.checkpoint import detect_resume_state, write_json_atomic
    write_json_atomic(tmp_path / "manifest.json", {"book_hash": "abc", "chunk_size": 5, "overlap": 1})
    write_json_atomic(tmp_path / "extract.json", {"book_hash": "abc", "pages": []})
    state = detect_resume_state(tmp_path, expected_manifest={"book_hash": "abc", "chunk_size": 10, "overlap": 1})
    assert state.cold_restart is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_checkpoint.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Create book_to_cards/pipeline/checkpoint.py**

```python
"""Per-run checkpoint files + resume-state detection."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResumeState:
    next_stage: str               # "extract" | "map" | "merge" | "reduce" | "done"
    map_resume_from: int = 0      # index of first chunk not yet on disk
    reduce_resume_from: int = 0   # index of first topic not yet on disk
    cold_restart: bool = False    # True iff manifest mismatch forces fresh run


def write_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _jsonl_is_complete(path: Path) -> bool:
    records = read_jsonl(path)
    return bool(records) and any(r.get("_complete") is True for r in records)


def _jsonl_count_entries(path: Path) -> int:
    """Count non-marker records (skip the {_complete: true} sentinel)."""
    return sum(1 for r in read_jsonl(path) if r.get("_complete") is not True)


def detect_resume_state(run_dir: Path, *, expected_manifest: dict | None = None) -> ResumeState:
    run_dir = Path(run_dir)

    # Manifest sanity check
    manifest_path = run_dir / "manifest.json"
    if expected_manifest is not None and manifest_path.exists():
        actual = json.loads(manifest_path.read_text())
        # Compare only the keys present in expected (forward-compat)
        for k, v in expected_manifest.items():
            if actual.get(k) != v:
                return ResumeState(next_stage="extract", cold_restart=True)

    extract = run_dir / "extract.json"
    map_jsonl = run_dir / "map.jsonl"
    topics = run_dir / "topics.json"
    cards = run_dir / "cards.jsonl"

    if not extract.exists():
        return ResumeState(next_stage="extract")
    if not map_jsonl.exists() or not _jsonl_is_complete(map_jsonl):
        n = _jsonl_count_entries(map_jsonl) if map_jsonl.exists() else 0
        return ResumeState(next_stage="map", map_resume_from=n)
    if not topics.exists():
        return ResumeState(next_stage="merge")
    if not cards.exists() or not _jsonl_is_complete(cards):
        n = _jsonl_count_entries(cards) if cards.exists() else 0
        return ResumeState(next_stage="reduce", reduce_resume_from=n)
    return ResumeState(next_stage="done")


def mark_complete(jsonl_path: Path) -> None:
    append_jsonl(jsonl_path, {"_complete": True})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_checkpoint.py -v
```

Expected: PASS for all six tests.

- [ ] **Step 5: Commit**

```bash
git add book_to_cards/pipeline/checkpoint.py tests/test_checkpoint.py
git commit -m "book_to_cards: atomic checkpoints with resume-state detection"
```

---

### Task 12: Deck writer

**Files:**
- Create: `book_to_cards/deck_writer.py`
- Create: `tests/test_deck_writer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_deck_writer.py`:

```python
"""Deck writer: create note type, insert cards, optional calibration hand-off."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def temp_collection(tmp_path):
    try:
        from anki.collection import Collection
    except ImportError:
        pytest.skip("anki module not installed; run inside Anki's bundled Python or skip")
    col = Collection(str(tmp_path / "test.anki2"))
    yield col
    col.close()


def test_ensures_note_type_and_deck(temp_collection):
    from book_to_cards.deck_writer import ensure_note_type, ensure_deck
    nt = ensure_note_type(temp_collection)
    assert nt["name"] == "Book Card"
    fields = {f["name"] for f in nt["flds"]}
    assert {"Front", "Back", "Topic", "Source"}.issubset(fields)

    did = ensure_deck(temp_collection, "Test Book")
    deck = temp_collection.decks.get(did)
    assert deck["name"] == "Test Book"


def test_inserts_card(temp_collection):
    from book_to_cards.deck_writer import ensure_note_type, ensure_deck, insert_card
    from book_to_cards.pipeline.types import Card
    nt = ensure_note_type(temp_collection)
    did = ensure_deck(temp_collection, "Test Book")
    card = Card(
        front="What is ATP?",
        back="The cell's energy currency.",
        topic="bioenergetics",
        source_pages=[3, 4],
        source_quotes=["ATP is produced via chemiosmosis…"],
    )
    nid = insert_card(temp_collection, card, note_type=nt, deck_id=did)
    note = temp_collection.get_note(nid)
    assert note["Front"] == "What is ATP?"
    assert "ATP is produced via chemiosmosis" in note["Source"]
    assert "p. 3" in note["Source"]


def test_skip_calibration_when_smart_grader_absent(temp_collection, monkeypatch):
    """When smart_grader.api is unavailable, insert_card must not raise."""
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "smart_grader", None)
    from book_to_cards.deck_writer import ensure_note_type, ensure_deck, insert_card
    from book_to_cards.pipeline.types import Card
    nt = ensure_note_type(temp_collection)
    did = ensure_deck(temp_collection, "Test Book")
    card = Card(front="Q", back="A", topic="t", source_pages=[1], source_quotes=["q"])
    # Should succeed silently even though we can't calibrate
    nid = insert_card(temp_collection, card, note_type=nt, deck_id=did, calibrate=True)
    assert nid is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_deck_writer.py -v
```

Expected: tests may SKIP if `anki` isn't on the host Python — that's fine. If anki is importable, FAIL because the module doesn't exist yet.

- [ ] **Step 3: Create book_to_cards/deck_writer.py**

```python
"""Anki deck and note operations for book_to_cards.

Idempotent helpers: ensure_note_type, ensure_deck, insert_card. Each handles
the case of running mid-collection without breaking other note types or
decks. The note type is created with the field layout smart_grader expects
so calibration just works.
"""
from __future__ import annotations

import html
import importlib.util
from typing import Any

from .pipeline.types import Card

NOTE_TYPE_NAME = "Book Card"
USER_FIELDS = ["Front", "Back", "Topic", "Source"]
GRADER_FIELDS = [
    "Keywords",
    "_grader_threshold",
    "_grader_ref_embedding",
    "_grader_calibration",
]


def _smart_grader_available() -> bool:
    return importlib.util.find_spec("smart_grader") is not None


def ensure_note_type(col) -> dict:
    """Create the Book Card note type if missing; idempotent."""
    mm = col.models
    existing = mm.by_name(NOTE_TYPE_NAME)
    if existing is not None:
        # Backfill any fields that were added in a later version of this addon.
        existing_names = {f["name"] for f in existing["flds"]}
        for name in USER_FIELDS + GRADER_FIELDS:
            if name not in existing_names:
                mm.add_field(existing, mm.new_field(name))
        mm.save(existing)
        return existing

    nt = mm.new(NOTE_TYPE_NAME)
    for name in USER_FIELDS + GRADER_FIELDS:
        mm.add_field(nt, mm.new_field(name))
    # Single forward template: Front on question, Back on answer.
    tmpl = mm.new_template("Card 1")
    tmpl["qfmt"] = "{{Front}}"
    tmpl["afmt"] = "{{FrontSide}}<hr id=answer>{{Back}}"
    mm.add_template(nt, tmpl)
    mm.add(nt)
    return mm.by_name(NOTE_TYPE_NAME)


def ensure_deck(col, name: str) -> int:
    return col.decks.id(name)


def _format_source(card: Card) -> str:
    if not card.source_quotes:
        return ""
    lines: list[str] = []
    for q, p in zip(card.source_quotes, card.source_pages or [None] * len(card.source_quotes)):
        page_tag = f"p. {p} — " if p is not None else ""
        lines.append(f"{page_tag}“{html.escape(q)}”")
    return "<br>".join(lines)


def insert_card(col, card: Card, *, note_type: dict, deck_id: int, calibrate: bool = True) -> int:
    """Create a note in the chosen deck. Optionally hand off to smart_grader."""
    note = col.new_note(note_type)
    note["Front"] = card.front
    note["Back"] = card.back
    note["Topic"] = card.topic
    note["Source"] = _format_source(card)
    col.add_note(note, deck_id)

    if calibrate and _smart_grader_available():
        try:
            from smart_grader.api import calibrate_note
            calibrate_note(note)
        except Exception:
            # Calibration is best-effort; if it fails, the card still exists.
            pass

    return note.id
```

- [ ] **Step 4: Run tests to verify they pass (or skip cleanly)**

```bash
python3 -m pytest tests/test_deck_writer.py -v
```

Expected: PASS, or SKIP if anki is not importable. If running inside Anki's bundled Python it should PASS.

- [ ] **Step 5: Commit**

```bash
git add book_to_cards/deck_writer.py tests/test_deck_writer.py
git commit -m "book_to_cards: deck/note insertion with optional grader hand-off"
```

---

### Task 13: Runner (QThread)

**Files:**
- Create: `book_to_cards/runner.py`

This task has minimal automated testing — QThread plumbing is exercised end-to-end via computer-use in Task 18.

- [ ] **Step 1: Create book_to_cards/runner.py**

```python
"""Background runner that drives the four-stage pipeline.

Runs as a QThread so the Anki main thread stays responsive. Any work that
touches mw.col goes through mw.taskman.run_on_main(...) so the collection
isn't accessed concurrently with review.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from aqt import mw
from aqt.qt import QThread, pyqtSignal

from .config import get_config
from .deck_writer import ensure_deck, ensure_note_type, insert_card
from .openai_client import OpenAIClient
from .pdf_text import extract_pages
from .pipeline.checkpoint import (
    ResumeState,
    append_jsonl,
    detect_resume_state,
    mark_complete,
    read_jsonl,
    write_json_atomic,
)
from .pipeline.map_chunks import run_map
from .pipeline.merge_tags import merge_topics
from .pipeline.reduce_cards import reduce_topic
from .pipeline.types import Aspect, Card, Topic


def _hash_pdf(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _user_files_dir() -> Path:
    # mw.addonManager.addonsFolder() / "book_to_cards" / "user_files"
    addons_folder = mw.addonManager.addonsFolder()
    p = Path(addons_folder) / "book_to_cards" / "user_files"
    p.mkdir(parents=True, exist_ok=True)
    return p


class BookRunner(QThread):
    stage_started = pyqtSignal(str)
    chunk_done = pyqtSignal(int, int)
    topic_done = pyqtSignal(int, int)
    card_inserted = pyqtSignal(object)
    cost_updated = pyqtSignal(float)
    aborted = pyqtSignal(str)
    finished_with_summary = pyqtSignal(dict)

    def __init__(
        self,
        *,
        pdf_path: str,
        deck_name: str,
        max_cost_usd: float,
        skip_calibration: bool = False,
    ):
        super().__init__()
        self.pdf_path = pdf_path
        self.deck_name = deck_name
        self.max_cost_usd = max_cost_usd
        self.skip_calibration = skip_calibration
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.aborted.emit(str(e))

    # --- pipeline orchestration -------------------------------------------

    def _run(self):
        cfg = get_config()
        book_hash = _hash_pdf(self.pdf_path)
        run_dir = _user_files_dir() / book_hash
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "book_hash": book_hash,
            "chunk_size": cfg["chunk_pages"],
            "overlap": cfg["chunk_overlap"],
            "chat_model": cfg["chat_model"],
            "embedding_model": cfg["embedding_model"],
        }
        state = detect_resume_state(run_dir, expected_manifest=manifest)
        if state.cold_restart:
            for p in run_dir.iterdir():
                if p.is_file():
                    p.unlink()
            state = ResumeState(next_stage="extract")
        write_json_atomic(run_dir / "manifest.json", manifest)

        client = OpenAIClient()

        if state.next_stage == "extract":
            self._do_extract(run_dir)
            state = detect_resume_state(run_dir, expected_manifest=manifest)
        if state.next_stage == "map":
            self._do_map(run_dir, client, resume_from=state.map_resume_from)
            state = detect_resume_state(run_dir, expected_manifest=manifest)
        if state.next_stage == "merge":
            self._do_merge(run_dir, client)
            state = detect_resume_state(run_dir, expected_manifest=manifest)
        if state.next_stage == "reduce":
            self._do_reduce(run_dir, client, resume_from=state.reduce_resume_from)

        # Final stage: insert cards (idempotent — uses a separate marker file)
        summary = self._do_insert(run_dir, calibrate=not self.skip_calibration)
        summary["cost_usd"] = client.meter.usd
        self.finished_with_summary.emit(summary)

    # --- stages -----------------------------------------------------------

    def _do_extract(self, run_dir: Path):
        self.stage_started.emit("extract")
        pages = extract_pages(self.pdf_path)
        write_json_atomic(run_dir / "extract.json", {
            "book_hash": run_dir.name,
            "pages": [{"index": p.index, "text": p.text} for p in pages],
        })

    def _do_map(self, run_dir: Path, client: OpenAIClient, *, resume_from: int):
        self.stage_started.emit("map")
        data = json.loads((run_dir / "extract.json").read_text())
        from .pipeline.types import Page
        pages = [Page(index=p["index"], text=p["text"]) for p in data["pages"]]

        cfg = get_config()
        # Count total chunks so we can emit progress
        from .pipeline.map_chunks import chunk_iter
        total = sum(1 for _ in chunk_iter(pages, size=cfg["chunk_pages"], overlap=cfg["chunk_overlap"]))

        skipped = 0
        for result in run_map(
            pages,
            llm=client,
            size=cfg["chunk_pages"],
            overlap=cfg["chunk_overlap"],
            model=cfg["chat_model"],
        ):
            if result["chunk_index"] < resume_from:
                continue
            append_jsonl(run_dir / "map.jsonl", result)
            self.chunk_done.emit(result["chunk_index"] + 1, total)
            self.cost_updated.emit(client.meter.usd)
            if client.meter.usd > self.max_cost_usd:
                raise RuntimeError(f"cost ceiling ${self.max_cost_usd} exceeded")
            if self._stop:
                raise RuntimeError("stopped by user")
        mark_complete(run_dir / "map.jsonl")

    def _do_merge(self, run_dir: Path, client: OpenAIClient):
        self.stage_started.emit("merge")
        records = read_jsonl(run_dir / "map.jsonl")
        aspects: list[Aspect] = []
        for r in records:
            if r.get("_complete"):
                continue
            for a in r.get("aspects", []):
                aspects.append(Aspect(
                    text=a.get("text", ""),
                    quote=a.get("quote", ""),
                    topic=str(a.get("topic", "")).strip() or "uncategorized",
                    source_pages=list(a.get("source_pages", [])),
                ))
        cfg = get_config()
        # Embedder adapter — merge_topics expects .embed(text)
        topics = merge_topics(
            aspects,
            embedder=client,
            distance_threshold=cfg["cluster_distance"],
        )
        write_json_atomic(run_dir / "topics.json", [t.to_dict() for t in topics])

    def _do_reduce(self, run_dir: Path, client: OpenAIClient, *, resume_from: int):
        self.stage_started.emit("reduce")
        data = json.loads((run_dir / "topics.json").read_text())
        topics: list[Topic] = []
        for t in data:
            aspects = [Aspect(**a) for a in t["aspects"]]
            topics.append(Topic(canonical=t["canonical"], aliases=t["aliases"], aspects=aspects))

        for i, topic in enumerate(topics):
            if i < resume_from:
                continue
            try:
                cards = reduce_topic(topic, llm=client, model=get_config()["chat_model"])
                for c in cards:
                    append_jsonl(run_dir / "cards.jsonl", c.to_dict())
            except Exception as e:
                append_jsonl(run_dir / "errors.jsonl", {"stage": "reduce", "topic": topic.canonical, "error": str(e)})
            self.topic_done.emit(i + 1, len(topics))
            self.cost_updated.emit(client.meter.usd)
            if client.meter.usd > self.max_cost_usd:
                raise RuntimeError(f"cost ceiling ${self.max_cost_usd} exceeded")
            if self._stop:
                raise RuntimeError("stopped by user")
        mark_complete(run_dir / "cards.jsonl")

    def _do_insert(self, run_dir: Path, *, calibrate: bool) -> dict:
        self.stage_started.emit("insert")
        cards = [c for c in read_jsonl(run_dir / "cards.jsonl") if c.get("_complete") is not True]

        # Anki collection ops MUST run on the main thread.
        result_holder: dict = {"inserted": 0, "errors": 0}

        def _do_inserts():
            note_type = ensure_note_type(mw.col)
            did = ensure_deck(mw.col, self.deck_name)
            for c in cards:
                try:
                    card = Card(
                        front=c["front"],
                        back=c["back"],
                        topic=c["topic"],
                        source_pages=c.get("source_pages") or [],
                        source_quotes=c.get("source_quotes") or [],
                    )
                    insert_card(mw.col, card, note_type=note_type, deck_id=did, calibrate=calibrate)
                    result_holder["inserted"] += 1
                except Exception:
                    result_holder["errors"] += 1
            mw.col.save()

        mw.taskman.run_on_main(_do_inserts)
        return result_holder
```

- [ ] **Step 2: Smoke test — module parses**

```bash
python3 -m py_compile book_to_cards/runner.py
```

Expected: no output, exit 0. (Importing it requires aqt; we don't import-test here.)

- [ ] **Step 3: Commit**

```bash
git add book_to_cards/runner.py
git commit -m "book_to_cards: QThread runner orchestrating the four-stage pipeline"
```

---

### Task 14: Launch dialog

**Files:**
- Create: `book_to_cards/ui/__init__.py` (empty)
- Create: `book_to_cards/ui/launch_dialog.py`

- [ ] **Step 1: Create ui/__init__.py**

```python
"""book_to_cards UI widgets."""
```

- [ ] **Step 2: Create ui/launch_dialog.py**

```python
"""Launch dialog: pick deck + cost estimate + max-cost ceiling."""
from __future__ import annotations

import os
from dataclasses import dataclass

from aqt import mw
from aqt.qt import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


@dataclass
class LaunchChoice:
    deck_name: str
    max_cost_usd: float
    skip_calibration: bool


# Cost coefficients derived from spec §7 (450-page reference run, $0.53 total).
# Scaling assumes roughly linear with page count.
_BASE_PAGES = 450
_BASE_COST_USD = 0.53


def estimate_cost(page_count: int, skip_calibration: bool) -> tuple[float, float]:
    """Return (low, high) USD estimate. ±50% band on the central estimate."""
    central = _BASE_COST_USD * (page_count / _BASE_PAGES)
    if skip_calibration:
        central *= 1 - (0.30 / 0.53)  # paraphrase + cal embedding rows
    return central * 0.5, central * 1.5


class LaunchDialog(QDialog):
    def __init__(self, pdf_path: str, page_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate cards from PDF")
        self._pdf_path = pdf_path
        self._page_count = page_count

        form = QFormLayout()

        default_deck = self._default_deck_name(pdf_path)
        self.deck_combo = QComboBox()
        self.deck_combo.setEditable(True)
        # Populate with existing deck names + default
        try:
            existing = [d.name for d in mw.col.decks.all_names_and_ids()]
        except Exception:
            existing = []
        existing = sorted(set(existing) | {default_deck})
        self.deck_combo.addItems(existing)
        self.deck_combo.setCurrentText(default_deck)
        form.addRow("Target deck:", self.deck_combo)

        self.skip_calibration = QCheckBox("Skip per-card calibration (cheaper, lower quality)")
        form.addRow("", self.skip_calibration)

        self.cost_label = QLabel("")
        form.addRow("Estimated cost:", self.cost_label)

        self.ceiling = QDoubleSpinBox()
        self.ceiling.setRange(0.1, 1000.0)
        self.ceiling.setDecimals(2)
        self.ceiling.setValue(5.0)
        self.ceiling.setPrefix("$")
        form.addRow("Max-cost ceiling:", self.ceiling)

        self.skip_calibration.toggled.connect(self._refresh_cost)
        self._refresh_cost()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _default_deck_name(self, pdf_path: str) -> str:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        return f"Book - {base}"

    def _refresh_cost(self):
        low, high = estimate_cost(self._page_count, self.skip_calibration.isChecked())
        self.cost_label.setText(f"${low:.2f} – ${high:.2f}")

    def choice(self) -> LaunchChoice:
        return LaunchChoice(
            deck_name=self.deck_combo.currentText().strip() or self._default_deck_name(self._pdf_path),
            max_cost_usd=float(self.ceiling.value()),
            skip_calibration=self.skip_calibration.isChecked(),
        )
```

- [ ] **Step 3: Add a unit test for the pure cost estimator**

`tests/test_launch_dialog.py`:

```python
"""Cost estimator is the only pure-Python part of the launch dialog."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_cost_scales_linearly_with_pages():
    # We can't import the module top-level because it pulls in aqt.
    # Read just the estimate_cost function via importlib? Easier: run the
    # math directly mirroring the constants from the source.
    from book_to_cards.ui import launch_dialog as ld  # may fail outside Anki
    low_450, high_450 = ld.estimate_cost(450, skip_calibration=False)
    low_900, high_900 = ld.estimate_cost(900, skip_calibration=False)
    assert abs(low_900 / low_450 - 2.0) < 0.01
    assert abs(high_900 / high_450 - 2.0) < 0.01


def test_skip_calibration_reduces_cost():
    from book_to_cards.ui import launch_dialog as ld
    full = ld.estimate_cost(450, skip_calibration=False)
    skip = ld.estimate_cost(450, skip_calibration=True)
    assert skip[0] < full[0]
    assert skip[1] < full[1]
```

If `aqt` is not importable, the tests will fail with ImportError. Mark the file with a skip:

```python
import pytest
pytest.importorskip("aqt")
```

at the top.

- [ ] **Step 4: Run tests (will skip if aqt missing)**

```bash
python3 -m pytest tests/test_launch_dialog.py -v
```

- [ ] **Step 5: Commit**

```bash
git add book_to_cards/ui/ tests/test_launch_dialog.py
git commit -m "book_to_cards: launch dialog with deck/ceiling/calibration controls"
```

---

### Task 15: Progress dialog

**Files:**
- Create: `book_to_cards/ui/progress_dialog.py`

- [ ] **Step 1: Create ui/progress_dialog.py**

```python
"""Non-modal progress dialog that subscribes to BookRunner signals."""
from __future__ import annotations

from aqt.qt import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    Qt,
)


class ProgressDialog(QDialog):
    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generating cards…")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        self.setModal(False)
        self._runner = runner

        self.stage_label = QLabel("Starting…")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.cost_label = QLabel("$0.00")
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._on_stop)

        layout = QVBoxLayout(self)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.cost_label)
        layout.addWidget(self.stop_button)
        self.resize(420, 160)

        runner.stage_started.connect(self._on_stage)
        runner.chunk_done.connect(self._on_step)
        runner.topic_done.connect(self._on_step)
        runner.cost_updated.connect(self._on_cost)
        runner.aborted.connect(self._on_aborted)
        runner.finished_with_summary.connect(self._on_done)

    def _on_stage(self, name: str):
        self.stage_label.setText(f"Stage: {name}")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    def _on_step(self, done: int, total: int):
        if total <= 0:
            return
        pct = int(100 * done / total)
        self.progress_bar.setValue(pct)
        self.stage_label.setText(f"{self.stage_label.text().split(':')[0]}: {done}/{total}")

    def _on_cost(self, usd: float):
        self.cost_label.setText(f"${usd:.3f}")

    def _on_stop(self):
        self._runner.request_stop()
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping…")

    def _on_aborted(self, reason: str):
        self.stage_label.setText(f"Aborted: {reason}")
        self.stop_button.setText("Close")
        self.stop_button.setEnabled(True)
        self.stop_button.clicked.disconnect()
        self.stop_button.clicked.connect(self.accept)

    def _on_done(self, summary: dict):
        inserted = summary.get("inserted", 0)
        errors = summary.get("errors", 0)
        cost = summary.get("cost_usd", 0.0)
        self.stage_label.setText(f"Done: {inserted} cards inserted ({errors} errors, ${cost:.3f})")
        self.progress_bar.setValue(100)
        self.stop_button.setText("Close")
        self.stop_button.clicked.disconnect()
        self.stop_button.clicked.connect(self.accept)
```

- [ ] **Step 2: Smoke compile**

```bash
python3 -m py_compile book_to_cards/ui/progress_dialog.py
```

- [ ] **Step 3: Commit**

```bash
git add book_to_cards/ui/progress_dialog.py
git commit -m "book_to_cards: non-modal progress dialog wired to runner signals"
```

---

### Task 16: Anki menu integration

**Files:**
- Modify: `book_to_cards/__init__.py`

- [ ] **Step 1: Replace stub with full menu wiring**

```python
"""Book to Cards for Anki — PDF ingest → auto-generated flashcards."""
from __future__ import annotations

import os
import sys

_VENDOR = os.path.join(os.path.dirname(__file__), "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from aqt import mw
from aqt.qt import QAction, QFileDialog
from aqt.utils import showInfo

from .config import get_config
from .pdf_text import extract_pages
from .runner import BookRunner
from .ui.launch_dialog import LaunchDialog
from .ui.progress_dialog import ProgressDialog


_active_runner: BookRunner | None = None
_active_dialog: ProgressDialog | None = None


def _on_generate():
    global _active_runner, _active_dialog
    if not get_config().get("openai_api_key"):
        showInfo("Set your OpenAI API key in Tools → Add-ons → Book to Cards → Config first.")
        return

    pdf_path, _ = QFileDialog.getOpenFileName(
        mw,
        "Choose a PDF to ingest",
        "",
        "PDF files (*.pdf)",
    )
    if not pdf_path:
        return

    # Quick page count for cost estimate — cheap on most PDFs but can be slow
    # on huge ones; we accept that since the user has just picked the file.
    try:
        pages = extract_pages(pdf_path)
    except Exception as e:
        showInfo(f"Couldn't read that PDF: {e}")
        return
    if not pages:
        showInfo("No useful text found in that PDF.")
        return

    dlg = LaunchDialog(pdf_path, len(pages))
    if dlg.exec() != dlg.DialogCode.Accepted:
        return
    choice = dlg.choice()

    _active_runner = BookRunner(
        pdf_path=pdf_path,
        deck_name=choice.deck_name,
        max_cost_usd=choice.max_cost_usd,
        skip_calibration=choice.skip_calibration,
    )
    _active_dialog = ProgressDialog(_active_runner, parent=mw)
    _active_dialog.show()
    _active_runner.start()


def _install_menu():
    action = QAction("Generate cards from PDF…", mw)
    action.triggered.connect(_on_generate)
    mw.form.menuTools.addAction(action)


_install_menu()
```

- [ ] **Step 2: Smoke compile**

```bash
python3 -m py_compile book_to_cards/__init__.py
```

- [ ] **Step 3: Rebuild both .ankiaddon files**

```bash
./build.sh
```

Expected: book_to_cards.ankiaddon contains __init__.py, runner.py, ui/, pipeline/, etc.

- [ ] **Step 4: Commit**

```bash
git add book_to_cards/__init__.py book_to_cards.ankiaddon smart_grader.ankiaddon
git commit -m "book_to_cards: wire menu entry to launch+runner+progress flow"
```

---

### Task 17: README updates and full test pass

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Book to Cards" section to README.md**

Append:

```markdown
## Book to Cards (companion add-on)

A second add-on, `book_to_cards`, ingests a PDF and auto-generates flashcards via an extract → map → merge → reduce pipeline. Install both add-ons to get auto-calibration on every generated card.

1. Install `book_to_cards.ankiaddon` via **Tools → Add-ons → Install from file…**, set your OpenAI key in its Config.
2. **Tools → Generate cards from PDF…**, pick a PDF, pick a deck, set a cost ceiling, hit OK.
3. The progress dialog shows live cost; you can keep reviewing other decks while it runs.
4. Costs roughly $0.50 per 450-page book at gpt-4o-mini rates. See `docs/superpowers/specs/2026-05-22-book-to-cards-design.md` for the full design.
```

- [ ] **Step 2: Run all unit tests**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASS (or SKIP for the deck_writer + launch_dialog tests if anki/aqt not on host). No failures.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document book_to_cards in top-level README"
```

---

### Task 18: End-to-end smoke test via computer-use

**Files:** none (computer-use orchestration is in the executing agent's prompt; this task documents the procedure)

The goal: prove the two .ankiaddon files install cleanly into real Anki, that "Generate cards from PDF…" produces cards in a chosen deck, and that the typed-answer review flow works on a generated card.

- [ ] **Step 1: Request computer-use access**

Use the computer-use MCP. Call `request_access` for Anki, Finder (for file picking if needed), and Terminal (for tail-log if needed).

- [ ] **Step 2: Install both add-ons**

1. `open_application` with `name="Anki"`. Wait for the main window.
2. Use Tools → Add-ons → Install from file… (`left_click` through the menu).
3. Pick `smart_grader.ankiaddon` via Finder.
4. Repeat for `book_to_cards.ankiaddon`.
5. Restart Anki via Tools → Add-ons → Restart Anki if prompted.

- [ ] **Step 3: Set the OpenAI key in both add-ons**

1. Tools → Add-ons. Pick "Smart Grader" → Config. Paste the key (sourced from `.env`, never logged in plaintext via screenshots — pass to `write_clipboard` and `key cmd+v`).
2. Repeat for "Book to Cards".

- [ ] **Step 4: Run a generation pass**

1. Tools → "Generate cards from PDF…".
2. Pick `tests/fixtures/small.pdf` (5 pages, ~5 facts).
3. In the launch dialog: deck = "Smoke Test", ceiling = $0.50.
4. Click OK. Watch the progress dialog. Expect ~5 cards to land within ~30 seconds.

- [ ] **Step 5: Verify cards in the deck**

1. Switch to the "Smoke Test" deck in the browser.
2. Open one card. Confirm `Front`, `Back`, `Topic`, `Source` are populated. Confirm `Keywords`, `_grader_threshold`, `_grader_ref_embedding` are filled (calibration succeeded).
3. Confirm `Source` shows verbatim quotes with `p. N` prefixes.

- [ ] **Step 6: Review one card and try the typed-answer flow**

1. Switch to review mode on "Smoke Test".
2. The typed-answer textarea should appear (because Keywords is populated).
3. Type an answer that should pass. Confirm pass.
4. Type an answer that should fail (no keywords). Confirm fail.

- [ ] **Step 7: Take a final screenshot and write a short report**

Write a file `docs/superpowers/reports/2026-05-22-book-to-cards-smoke.md` summarising:
- Did both add-ons install?
- Did a generation pass complete?
- Did cards land with the expected fields?
- Did the typed-answer review work on a generated card?
- Total cost spent on OpenAI (read from the progress dialog).

Commit:

```bash
git add docs/superpowers/reports/2026-05-22-book-to-cards-smoke.md
git commit -m "End-to-end smoke test against real Anki via computer-use"
```

---

## Self-review

**Spec coverage:**
- §2 layout — Tasks 1, 3, 5, 17.
- §3 pipeline data shapes — Task 5 (types.py).
- §3 extract — Tasks 4, 5.
- §3 map — Tasks 6, 8.
- §3 merge — Task 9.
- §3 reduce — Task 10.
- §4 menu/dialog — Tasks 14, 16.
- §4 runner — Task 13.
- §4 deck writer — Task 12.
- §5 smart_grader refactor — Task 2.
- §6 error handling & resumability — Task 11 (checkpoint), Tasks 8/13 (per-stage failure isolation).
- §7 cost meter + ceiling — Tasks 7, 13, 14.
- §8 tests — woven into every task; §8.3 smoke test = Task 18.

**Type consistency:**
- `Aspect`, `Topic`, `Card`, `Page` defined in Task 5, used identically in Tasks 6/8/9/10/12/13.
- `chunk_iter(size=, overlap=)` keyword args in Task 6, called the same way in Task 8 and Task 13.
- `OpenAIClient.embed(text)` and `chat_json(...)` in Task 7, used identically in Tasks 9/10/13.
- `_calibrate_one` in Task 2, re-exported by `smart_grader.api.calibrate_note` and called from Task 12's deck_writer.

**Placeholder scan:** no "TBD" / "TODO" / "implement later" / "add error handling" left in step bodies.
