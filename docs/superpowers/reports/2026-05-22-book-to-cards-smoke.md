# Book-to-Cards end-to-end smoke test (Task 18)

**Date:** 2026-05-22
**Anki:** 25.02.4 (Python 3.9.18, PyQt 6.6.1, macOS arm64)
**Input:** `tests/fixtures/small.pdf` — 5-page born-digital PDF (biology snippets)
**Driver:** computer-use MCP + osascript for menu interactions
**Outcome:** **Pass.**

## Steps executed

1. **Installed both add-ons** via Tools → Add-ons → Install from file → `smart_grader.ankiaddon` and `book_to_cards.ankiaddon`.
2. **Configured OpenAI API key** in each add-on's Config (key sourced from `.env`, never logged).
3. **Restarted Anki** to load the freshly-installed code.
4. **Triggered Tools → Generate cards from PDF…**, selected `tests/fixtures/small.pdf`.
5. The Launch Dialog showed:
   - Target deck: `Book - small` (auto-generated from filename)
   - Estimated cost: `$0.00 – $0.01`
   - Ceiling: `$5.00`
6. Pressed OK. Progress dialog reported four stages (`extract`, `map`, `merge`, `reduce`) over ~75 seconds.
7. Final dialog reported: **"Done: 8 cards inserted (0 errors, $0.001)"**.

## Issues fixed during the smoke run

- **PEP 604 syntax in pdfminer.six.** The latest pdfminer.six release uses `bytes | str` and `dict[str, int | bool]` at module load. Anki ships Python 3.9, which doesn't support that syntax outside `from __future__ import annotations`. Two part fix:
  1. Downgrade vendored pdfminer to `20231228` (last Python-3.9-compatible release).
  2. Prepend `from __future__ import annotations` to every .py file in `book_to_cards/_vendor/pdfminer/` for defence in depth.

  Module-level type aliases in `jbig2.py` (e.g. `JBIG2SegmentFlags = dict[str, int | bool]`) had to come from the older version — `__future__` annotations only defer annotation evaluation, not runtime expressions.
- **Optional cryptography import.** Patched `pdfminer/pdfdocument.py` to wrap the `cryptography` imports in `try/except` so the package loads without the C-extension dep. Encrypted PDFs would now fail at runtime with `AttributeError` rather than at import time, which is the right tradeoff for the textbook use case.

## Verification

Queried Anki's SQLite collection directly. The eight newly-inserted notes each carry every expected field:

| field | sample value |
|---|---|
| `Front` | "How is ATP produced during the light reactions of photosynthesis?" |
| `Back` | "ATP is produced via chemiosmosis across the thylakoid membrane during the light reactions." |
| `Topic` | "ATP Production" |
| `Source` | `p. 2 — "ATP is produced via chemiosmosis across the thylakoid membrane during the light reactions."` |
| `Keywords` | "ATP, chemiosmosis, thylakoid, membrane, light reactions" |
| `_grader_threshold` | `0.9037` (calibrated per card; observed range 0.78–0.92) |
| `_grader_ref_embedding` | populated (1536-dim JSON float array) |

This confirms:

- **PDF extraction** worked on the fixture.
- **Map stage** emitted aspects with verbatim quotes and topic strings.
- **Merge stage** clustered topics correctly (multiple aspects ended up under the canonical `Mitochondria` topic).
- **Reduce stage** produced atomic, single-fact cards grounded in the source quotes.
- **Deck writer** populated all four user-facing fields.
- **Smart Grader hand-off** ran: per-card `Keywords` auto-generated, `_grader_threshold` calibrated, `_grader_ref_embedding` persisted.

## Cost

- **Reported by the addon:** $0.001 for the whole run (5 pages → 8 cards).
- Consistent with the spec's $0.53 / 450-page estimate scaled down.

## Known gaps (out of scope for the smoke)

- Did not exercise the typed-answer review flow on a generated card (the progress dialog held focus and made the Browse window navigation awkward through the computer-use driver). Calibration data is on the notes, so the typed-answer UI should activate when the deck is reviewed — but this is currently asserted by inspection of the DB rather than by clicking through the review.
- Did not test cost-ceiling abort path (the run cost $0.001 against a $5 ceiling; never close to triggering).
- Did not test resumability (the run completed in one shot).

## Verdict

Both add-ons install cleanly, the pipeline runs end-to-end against a real PDF, and Smart-Grader-ready cards land in the chosen deck. **Task 18 passes.**
