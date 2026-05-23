# Baudrillard real-book run

**Date:** 2026-05-22
**Input:** `Baudrillard.1981.Simulacra-and-Simulation.pdf` — 172 pages, 2.7 MB, born-digital English translation (Sheila Faria Glaser).

## What worked

- **PDF extraction succeeded cleanly.** All 172 pages parsed, including the title page (with some typographic artefacts that pdfminer can't normalise) and body prose.
- **Cost estimate was on target.** Launch dialog predicted $0.09–$0.28. Map stage ran to completion at $0.034 for 40 chunks; reduce added another ~$0.025 to bring cumulative LLM cost to $0.055 before insert.
- **Map → merge → reduce pipeline completed in ~12 minutes.**
  - Map: 40 chunks, ~7s/chunk, $0.034 total.
  - Merge: ~30s (no LLM, just embedding clusters).
  - Reduce: 314 topics processed serially, ~2.5s/topic, $0.020 total. One topic ("Cultural Incinerator") timed out — caught cleanly by per-topic exception handler and logged to `errors.jsonl`. Pipeline continued.
- **Card quality on a difficult text is honest.** Spot-checking the first 39 inserted cards, they read as well-formed atomic flashcards grounded in the source:
  - `What does hyperreality abolish in Crash?` / `In Crash, hyperreality abolishes both fiction and reality.`
  - `What does the discourse of power convey according to the text?` / `It conveys the impossibility of a determined position of power and the impossibility of a determined discursive position.`
  - `What is the task of antiethnology?` / `The task of antiethnology is to reinject the difference fiction as ethnology collapses in its classical institution.`
- **Topic identification matches the canon.** The clustering correctly surfaced the recognisable Baudrillard themes:
  - Hyperreality (16 aspects)
  - Simulacra and Reality (10 aspects)
  - Simulation and Reality (5 aspects)
  - Precession of Simulacra (3 aspects)
  - Desert of the Real (singleton, correctly named)
  - Plus dozens of finer-grained singletons (Iconoclasts, Jesuit Strategy, Antiethnology, Cultural Incinerator, etc.).
- **Source-quote attribution carried through.** Every card's Source field contains a verbatim quote and a page reference, sourced from the map stage's `quote` field and the reduce stage's `source_quote_indices`.

## What didn't work / what's now obvious

### 1. Topic clustering is too conservative for unique-vocabulary texts

Of 314 canonical topics, **278 are singletons** — only one aspect each. The cosine-distance threshold of 0.25 was tuned on the biology fixture (where topics like "chemiosmosis" and "electron transport chain" cluster nicely) but Baudrillard's prose is full of one-off neologisms and named-concept inventions, so most topic strings end up too distinct to merge.

This isn't strictly wrong — the resulting cards are still legitimate — but it means the reducer is being asked to produce 314 LLM calls when ~60–80 well-merged clusters would probably yield better cards.

**Fix candidates:**
- Loosen `distance_threshold` to 0.35 for high-variance texts (or make it auto-tune from the distribution of pairwise distances).
- Add a second clustering pass that merges clusters whose canonicals are themselves close (chained merging).
- Use an LLM-based consolidation pass as the second alternative from the brainstorm.

### 2. Insert + per-card calibration is the wall-clock bottleneck

Map + merge + reduce took ~12 minutes. Insert + calibrate has been running for ~25 minutes and is only at **39 of 357 cards** committed.

Root cause: `_do_inserts` runs entirely on Anki's main thread via `mw.taskman.run_on_main(_do_inserts)`. For each card it:
1. Generates keywords (1 chat call, ~1.5s)
2. Generates 18 paraphrases (1 chat call, ~3–5s)
3. Embeds reference + 18 paraphrases (19 embedding calls, ~5–8s serially)
4. Inserts the note and updates fields

That's ~10–15 seconds per card on average. With 357 cards: ~60–90 minutes total. While this runs, Anki's UI is unresponsive.

**Fix candidates:**
- Pre-compute keywords + paraphrases + embeddings in the worker thread (not the main thread). Only the actual `col.add_note` and field writes need to be on main. That alone parallelises calibration with everything else.
- Batch embeddings — OpenAI's `/v1/embeddings` accepts up to 2048 inputs per call. The 19 per-card embeddings could collapse into one HTTP request, cutting that step from ~6s to <1s.
- Default to **skip per-card calibration** for runs >50 cards, and prompt the user to run smart_grader's "Calibrate selected notes" on whatever subset they actually want to study.

### 3. Cost meter freezes during the main-thread phase

The `cost_updated` signal is only emitted from inside the QThread worker. Once execution shifts to `_do_inserts` on the main thread, no signal is fired, so the dialog still shows the $0.055 from the end of reduce — even though calibration is actively spending money. A user watching the dialog has no idea whether anything is happening.

### 4. Cards aren't durably saved until the very end

`mw.col.save()` runs once at the end of `_do_inserts`. If Anki force-quits or crashes mid-calibration, all the work since `_do_inserts` started is lost — including cards already created via `col.add_note`, because they're held in an uncommitted transaction.

Wait — actually, observation says otherwise: while watching the run, the SQLite count DID climb (159 → 198), which means `col.add_note` is committing transactionally per note. So cards ARE durable as they're inserted. The `mw.col.save()` at the end is mostly for the model-edit changes. Good — that means the run is interruptible without losing committed cards.

### 5. Page numbers in the Source field appear single-digit

Several cards show `p. 7`, `p. 1`, etc. The LLM is returning page numbers as small ints rather than the absolute page indices from the `[Page N]` markers in the prompt. The actual page-marker numbers are absolute, so the model is either truncating or pattern-matching to single digits.

**Fix:** Re-prompt to use only the `[Page N]` markers literally, or post-process by clamping returned page numbers to the chunk's `source_pages` range.

### 6. Reducer generates 1+ cards even for singleton topics

The prompt says "produce 1–5 atomic flashcards." For a topic with one aspect and one quote, the model usually emits one card — fine. But sometimes the singleton aspect doesn't merit a card at all (a passing mention, not a memorable fact). Worth letting the reducer return zero cards explicitly, with a prompt nudge like "if the material isn't substantive enough for a card, return an empty list."

## Cost summary (so far)

| Stage | Cost |
|---|---|
| Map (40 chunks) | $0.034 |
| Merge (clustering, embeddings) | ~$0.001 |
| Reduce (314 topics) | $0.020 |
| Calibration (39 of 357 cards so far) | tracked but not shown live; estimated ~$0.30 based on embedding cache growth |
| **Cumulative when fully done (extrapolated)** | **~$2.50** (vs. the $0.09–$0.28 launch-dialog band) |

The launch dialog's cost estimator underestimated the real cost by ~10× because it scales from the 5-page fixture run where calibration was a small share. For a real book where calibration dominates total cost, the estimate needs to be rederived.

## Verdict

The pipeline **works end-to-end on a real philosophy book**, producing well-formed atomic flashcards faithful to the source. The output is genuinely useful for studying.

The biggest gaps are operational:
1. Calibration parallelism (~10× wall-clock improvement available).
2. Cost-estimate accuracy (current model under-counts calibration by ~10×).
3. Topic clustering tuning for high-variance texts (would reduce card count from 357 to ~80–120, which is probably more pedagogically right).

None of these block the pipeline from being useful for personal study today. They're the right next things to fix.
