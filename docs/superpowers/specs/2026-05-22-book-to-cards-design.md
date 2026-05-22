# Book-to-Cards: PDF → Anki flashcards via map-merge-reduce

**Date:** 2026-05-22
**Status:** approved (pending user review of this document)

## 1. Goal

A second Anki add-on that ingests a PDF (e.g. a 450-page textbook) and produces well-formed flashcards in a target deck, fully automated and runnable in the background while the user keeps reviewing.

The existing `smart_grader` add-on is reused as-is; generated cards are auto-calibrated through `smart_grader.calibrate_note` after insertion so they become gradable via the typed-answer flow.

## 2. Architecture

### 2.1 Two-add-on layout

```
Anki-Plugin/
├── build.sh                       # builds BOTH .ankiaddon files
├── smart_grader/                  # existing; minor refactor (§5)
├── book_to_cards/                 # new
│   ├── __init__.py                # Anki hooks: menu item, runner glue
│   ├── manifest.json
│   ├── config.json
│   ├── config.py
│   ├── openai_client.py           # vendored copy of smart_grader's
│   ├── pdf_text.py                # pdfminer wrapper → list[Page]
│   ├── pipeline/
│   │   ├── extract.py
│   │   ├── map_chunks.py
│   │   ├── merge_tags.py
│   │   ├── reduce_cards.py
│   │   └── checkpoint.py
│   ├── runner.py                  # QThread driving the pipeline
│   ├── ui/
│   │   ├── launch_dialog.py
│   │   └── progress_dialog.py
│   ├── deck_writer.py
│   └── _vendor/pdfminer/          # bundled pure-Python extractor
└── docs/superpowers/specs/2026-05-22-book-to-cards-design.md
```

Each `.ankiaddon` is independent. The OpenAI client is duplicated rather than shared (Anki add-ons cannot reliably depend on each other at import time). The two add-ons each take their own `openai_api_key` in config; the user enters it twice.

### 2.2 Why two add-ons, not one

- Either can be enabled/disabled independently.
- Each gets its own Anki config screen, no shared state.
- Avoids fragile cross-add-on imports. The only optional cross-talk is `book_to_cards` calling `smart_grader.api.calibrate_note` after card insertion if and only if `importlib.util.find_spec("smart_grader")` returns truthy.

## 3. Pipeline

Each stage takes one well-defined input, writes one JSON-serialisable output to disk, then exits. The output of stage N is the only input to stage N+1; resume logic (§6) keys on what files are present.

### 3.1 Data shapes

```python
@dataclass
class Page:
    index: int          # 0-based, matches PDF page numbering
    text: str           # whitespace-collapsed pdfminer output

@dataclass
class Aspect:
    text: str           # one-sentence distillation of the fact
    quote: str          # verbatim excerpt from the source pages
    topic: str          # single model-chosen topic string
    source_pages: list[int]

@dataclass
class Topic:
    canonical: str               # cluster centroid's tag string
    aliases: list[str]           # other tag strings in the cluster
    aspects: list[Aspect]        # every aspect whose topic ∈ aliases

@dataclass
class Card:
    front: str
    back: str
    topic: str
    source_pages: list[int]
    source_quotes: list[str]     # contributing quotes, surfaced on the Source field
```

### 3.2 Extract

`pdf_text.extract(pdf_path) -> list[Page]` — calls vendored pdfminer.six, collapses runs of whitespace, drops pages with < 50 non-whitespace characters (PDF artefacts, blank scan pages). Writes `extract.json`:

```json
{ "book_hash": "sha256:…", "pages": [Page, …] }
```

### 3.3 Map

Yields sliding chunks: pages `[0..4]`, `[4..8]`, `[8..12]`, … i.e. 5 pages per chunk with 1-page overlap. The overlap catches concepts that straddle a chunk boundary; the offset of 4 means each page (except the first and last) appears in exactly two chunks.

For each chunk, one LLM call to `gpt-4o-mini`:

```
System:
You are an expert at extracting study material from textbooks. For the
passage below, list every noteworthy fact, definition, mechanism, or
claim a student should remember. For each item, return:
  - "text": a one-sentence distillation
  - "quote": the exact verbatim excerpt from the passage (no edits)
  - "topic": a short topic name (1-4 words)
  - "source_pages": which page numbers (from the page markers below) the
    excerpt came from
Respond with a single JSON object: {"aspects": [...]}. No prose, no fences.

User:
[Page 47]
<text of page 47>
[Page 48]
<text of page 48>
...
```

Response is appended to `map.jsonl` — one chunk per line — atomically (write to tmp, rename). A crash mid-map loses at most the in-flight chunk.

### 3.4 Merge

No LLM calls. Steps:

1. Collect every distinct `aspect.topic` string across all chunks.
2. Embed each via `text-embedding-3-small`.
3. Agglomerative clustering with cosine distance threshold = 0.25. The threshold is conservative: tighter than typical semantic-similarity uses, but tags are short strings where small differences matter ("oxidative phosphorylation" ≠ "phosphorylation").
4. Each cluster's centroid-nearest tag string becomes the canonical name.
5. Every aspect whose `topic` falls in a cluster gets bundled into that cluster's `Topic.aspects`.

Output: `topics.json` → `[Topic, …]`. Cheap, deterministic, fully testable without API access.

### 3.5 Reduce

For each `Topic`, one LLM call to `gpt-4o-mini`:

```
System:
You are writing Anki flashcards from study material on one topic. You
will be given the topic name, its aliases, and several extracted facts
(each with a verbatim source quote). Produce 1-5 atomic flashcards.

Rules:
- Each card tests exactly one fact. If a topic has multiple distinct
  facts, emit multiple cards.
- Ground every card in the provided quotes. Do not introduce information
  not supported by them.
- Front is a question. Back is the answer in 1-3 sentences.
- Don't include the topic name in the front of the card unless it's the
  natural way to phrase the question.

Respond with a single JSON object:
{"cards": [{"front": "...", "back": "...", "source_quote_indices": [0, 2]}]}

User:
Topic: oxidative phosphorylation
Aliases: chemiosmosis, electron transport chain
Quotes:
  [0] "...the proton gradient established by the electron transport chain
       drives ATP synthase, producing ATP through a process called
       chemiosmosis..." (pp. 49)
  [1] "Oxidative phosphorylation occurs across the inner mitochondrial
       membrane..." (pp. 75)
  [2] ...
```

The reduce step resolves `source_quote_indices` back to the actual quote strings before persisting, so the `Card` dataclass on disk carries `source_quotes: list[str]` directly — the indices are a transport detail of the LLM response, never stored. Output: `cards.jsonl`.

## 4. Anki integration

### 4.1 Menu and dialog flow

`Tools → Generate cards from PDF…` opens a file picker. After selection, `launch_dialog.py` shows:

- **Target deck** — combo box, default = new deck named after PDF filename (sanitised).
- **Note type** — defaults to a `Book Card` type the add-on creates on first run, with fields: `Front`, `Back`, `Topic`, `Source`, `Keywords`, `_grader_threshold`, `_grader_ref_embedding`, `_grader_calibration`. The first four are user-facing; the last four are managed by `smart_grader`.
- **Cost estimate** — range, derived from the §7 numbers scaled by `page_count / 450`. Shown as a low / high band (±50% to reflect topic-count variance).
- **Max-cost ceiling** — spinner, default $5.
- **Skip per-card calibration** — checkbox. Off by default. When on, generated cards skip the paraphrase step and use the global fallback threshold.
- **Run in background** button — kicks off `runner.start()` and shows the progress dialog.

### 4.2 Background runner

`runner.py` is a `QThread` subclass. It owns the pipeline and emits Qt signals: `stage_started(name)`, `chunk_done(index, total)`, `topic_done(index, total)`, `card_inserted(card)`, `cost_updated(usd)`, `aborted(reason)`, `finished(summary)`.

The progress dialog subscribes; it's non-modal so review continues uninterrupted. The runner uses Anki's `mw.taskman.run_on_main(...)` for any collection writes — short bursts on the main thread, the rest of the work on the worker.

Card insertion happens incrementally: every 5 cards from the reduce step trigger a `run_on_main(insert_batch)` callback that writes them to the collection. The user sees the deck fill up as topics finish.

### 4.3 Deck writer

`deck_writer.py`:

1. Ensures the `Book Card` note type exists (idempotent).
2. Ensures the target deck exists.
3. For each card: creates a note, sets `Front` / `Back` / `Topic` / `Source` (the `Source` field renders each quote on its own line, preceded by `p. N`).
4. If `smart_grader.api.is_available()`: calls `smart_grader.api.calibrate_note(note)` immediately. Otherwise leaves the grader fields empty and shows a single tooltip at finish ("Install Smart Grader to enable typed-answer grading on these cards").

### 4.4 Cost meter

`openai_client.py` records `(input_tokens, output_tokens)` from every API response into a shared counter. The runner reads it after each call and aborts cleanly if `cumulative_cost > ceiling`. Partial progress is preserved on disk (§6); the dialog gets a "raise ceiling and resume" button.

## 5. `smart_grader` refactor (minimal)

Two changes:

1. **New `smart_grader/api.py`** — re-exports `calibrate_note(note) -> dict` and `is_available() -> bool`. This is the only surface `book_to_cards` imports.
2. **Extract `_calibrate_one(note)`** out of `calibrate_selected_notes` so the per-note work can be invoked without spawning the existing progress dialog. The existing menu-driven flow keeps its dialog; external callers (like `book_to_cards`'s own progress dialog) call `_calibrate_one` directly.

No behaviour change for existing `smart_grader` users.

## 6. Error handling and resumability

### 6.1 Checkpoint layout

Per run: `<addon_dir>/user_files/<book_hash>/`

```
extract.json
map.jsonl                   # appended one line per chunk; final {"_complete": true} when done
topics.json
cards.jsonl                 # appended; final {"_complete": true} when done
errors.jsonl                # appended whenever a stage skips an item
manifest.json               # {book_hash, chunk_size, overlap, models, ceiling}
```

`book_hash` is `sha256(pdf_bytes)`. `manifest.json` is written at launch; on relaunch a mismatch in chunk size / overlap / model names triggers a cold restart with a confirmation dialog.

### 6.2 Resume rules

| Files present | Action |
|---|---|
| Nothing | Start at extract. |
| `extract.json` | Skip extract; start map. |
| Partial `map.jsonl` | Resume map at the first chunk index not yet on disk. |
| `map.jsonl` with `_complete` marker | Skip extract+map; start merge. |
| `topics.json` | Skip merge; start reduce. |
| Partial `cards.jsonl` | Resume reduce at the first topic not yet emitted. |

Atomic writes everywhere: `write to *.tmp; os.replace(*.tmp, dest)` for whole files; for append-only `.jsonl`, `open(path, "a")` + per-line `f.write` + `f.flush()` is sufficient (POSIX guarantees small append atomicity).

### 6.3 Per-stage failure handling

- **Extract** — pdfminer raises on malformed PDF. Hard error; nothing to resume from.
- **Map per-chunk** — exceptions caught, chunk index logged to `errors.jsonl`, processing continues. End-of-map dialog summary reports skipped chunks.
- **Merge embedding call** — failures logged; the unembeddable tag string goes into a singleton "uncategorised" cluster.
- **Reduce per-topic** — exceptions caught, topic logged, continue.
- **Anki write failures** — caught per-note; note ID + error to `errors.jsonl`; runner keeps going.
- **Cost ceiling hit** — clean abort, status `paused_over_budget`; relaunching with the same PDF resumes from the latest checkpoint.
- **User closes Anki mid-run** — worker thread killed at app exit. Next launch + same PDF resumes from latest checkpoint.

## 7. Cost (back-of-envelope)

For a 450-page book on OpenAI:

| Stage | Model | Cost |
|---|---|---|
| Map (90 chunks) | gpt-4o-mini | ~$0.12 |
| Tag embedding | text-embedding-3-small | ~$0.0001 |
| Reduce (~100 topics) | gpt-4o-mini | ~$0.08 |
| Auto-keywords (~300 cards) | gpt-4o-mini | ~$0.03 |
| Paraphrases (~300 cards) | gpt-4o-mini | ~$0.29 |
| Calibration embeddings | text-embedding-3-small | ~$0.01 |
| **Total** | | **~$0.53** |

The "skip per-card calibration" checkbox shaves off the paraphrase + calibration-embedding rows (~$0.30), bringing the total to ~$0.23 — useful for first-pass exploration before committing to the deck.

Default ceiling: **$5** (10× safety margin).

## 8. Testing

### 8.1 Unit, no Anki / no network

- **`pdf_text`** — checked-in fixture PDFs (born-digital, multi-column, scan-with-blanks). Asserts page count and known substrings.
- **`map_chunks.chunk_iter()`** — pure function `pages → chunks`. Property tests: every page appears in ≥1 chunk; consecutive chunks overlap by exactly 1; tail handling.
- **`merge_tags.cluster()`** — hand-built `(tag, vector)` lists with known cosine structure; asserts cluster membership.
- **`reduce_cards.build_prompt()`** — snapshot test on the prompt string.
- **`checkpoint`** — round-trip read/write + atomic-replace behaviour; parameterised test over the resume-rule table from §6.
- **LLM/embedding clients** — protocol-based mocks return canned JSON; the pipeline modules are exercised end-to-end without network.

### 8.2 Integration, headless Anki, no network

- **`deck_writer`** — `anki.collection.Collection(temp_path)` against a temp `.anki2` file. Asserts that a fake card payload becomes a real note with the right fields, deck, and note type. Asserts that when `smart_grader` is absent, fields it would have filled stay empty.

### 8.3 End-to-end via computer-use

A one-time smoke test, repeated whenever the addon's surface changes:

1. Build both `.ankiaddon` files via `./build.sh`.
2. Open Anki via computer-use.
3. Install both add-ons via `Tools → Add-ons → Install from file…`.
4. Enter the OpenAI key in both configs (sourced from local `.env`, never logged).
5. Drop a short fixture PDF (~10 pages) on the menu picker.
6. Watch the progress dialog. Assert cards arrive in the target deck.
7. Switch to review on that deck. Type an answer. Assert the typed-answer UI shows pass/fail.
8. Inspect one card in the browser; assert `Source` field has page numbers and verbatim quotes.

This is run manually (not in CI) because it needs a real OpenAI key and a desktop Anki install.

## 9. Out of scope (deferred)

- Image/figure extraction from PDFs. Text only.
- OCR fallback for scanned PDFs. The fixture set is born-digital.
- Multi-PDF runs in one go. One PDF per invocation.
- Custom prompt overrides per-deck. Single global prompt template.
- Live editing of generated cards before insertion. Cards land in the deck; the user edits with Anki's normal browser.
- Re-running on a partially-edited deck (i.e. "regenerate this topic"). Resumability is at the run level, not the topic level.

## 10. Open questions

None — all resolved during brainstorming.
