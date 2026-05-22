# Smart Grader for Anki

Type your answer into a textbox on the front of the card. The add-on checks:

1. **Keyword gate** (strict AND) — every required keyword must appear as a whole word.
2. **Semantic similarity** — cosine similarity between your answer's embedding and the reference answer's embedding (OpenAI `text-embedding-3-small`).
3. **Per-card threshold** — calibrated automatically by asking GPT to generate good/bad paraphrases of the reference answer and fitting the threshold to the resulting similarity distribution.

You still grade yourself with Again/Hard/Good/Easy. The add-on shows pass/fail and a diff; the SRS decision is yours.

## Install

1. Close Anki.
2. Install `smart_grader.ankiaddon` via **Tools → Add-ons → Install from file…**, or drop the `smart_grader/` source tree into your Anki add-ons directory:
   - macOS: `~/Library/Application Support/Anki2/addons21/`
   - Linux: `~/.local/share/Anki2/addons21/`
   - Windows: `%APPDATA%\Anki2\addons21\`
3. Start Anki, go to **Tools → Add-ons → Smart Grader → Config**, paste your OpenAI API key.

## Set up your note type

All you need on the note is a question field (`Front`, `Question`, or `Prompt`) and an answer field (`Back`, `Answer`, or `Definition`). The add-on creates the `Keywords` field for you the first time you calibrate, and fills it in by asking GPT which terms a correct answer must contain.

If you want to override the AI-picked keywords for a particular card, just type your own into the `Keywords` field — calibration will respect anything already there. Clear the field to regenerate.

`_grader_threshold`, `_grader_ref_embedding`, and `_grader_calibration` are also created automatically the first time you calibrate.

## Calibrate

1. Open the **Browse** window. Select the notes you want to calibrate.
2. **Tools → Smart Grader: Calibrate selected notes**.
3. Wait — each note costs one GPT-mini keyword call (skipped if Keywords is already filled), one GPT-mini paraphrase call, and ~20 embeddings.

What it does, per note:

1. **Keywords** — if the `Keywords` field is empty, ask GPT for the 3–6 load-bearing terms a correct answer must contain, and save them to the note.
2. **Paraphrases** — ask GPT for 12 valid rewordings of the answer and 6 plausible-but-wrong near-misses.
3. **Threshold** — embed everything, compute cosine to the reference, and set

       threshold = max( min(similarity of good paraphrases),  max(similarity of bad ones) + 0.01 )

   So the threshold sits at the floor of the "good" distribution, but is pushed up if any "bad" paraphrase managed to score higher (which would mean a global threshold would be unsafe for this card).
4. **Reference embedding** — the reference's vector is saved to `_grader_ref_embedding` so review never has to recompute it.

Mean/stddev and the raw numbers are stored in `_grader_calibration` for inspection. Uncalibrated cards fall back to `0.82` (configurable).

## Costs

At early-2026 OpenAI pricing, calibrating one card runs roughly $0.001–$0.003 (mostly GPT-mini tokens). Reviewing a card is **one** embedding — the user's answer — because the reference vector is stored on the note. Embeddings are also cached on disk by SHA256, so re-typing an identical answer is free.

## Building from source

```
./build.sh
```

Recompiles every module under `smart_grader/` and rebuilds `smart_grader.ankiaddon` with files at the zip root (Anki rejects nested layouts).

## Config

`Tools → Add-ons → Smart Grader → Config`:

| key | default | what it does |
|---|---|---|
| `openai_api_key` | `""` | required |
| `embedding_model` | `text-embedding-3-small` | swap for `-large` if you want |
| `chat_model` | `gpt-4o-mini` | for paraphrase generation |
| `n_good_paraphrases` | `12` | more = tighter calibration, more cost |
| `n_bad_paraphrases` | `6` | |
| `fallback_threshold` | `0.82` | used when a card isn't calibrated yet |
| `epsilon_above_bad` | `0.01` | margin enforced over the worst "bad" paraphrase |

## When calibration is unreliable

If the LLM's "bad" paraphrases end up *more* similar to the reference than the worst "good" paraphrase, the two distributions overlap. The diagnostics blob records `"overlap_warning": true` for those cards. Usually it means the reference answer is too short to distinguish from near-misses by embedding distance alone — consider expanding the answer or relying more on the keyword gate.
