"""
Per-card calibration.

For each note:
    1. Generate ~N_good paraphrases that should pass + ~N_bad near-misses that
       should fail. (LLM-generated; see openai_client.generate_paraphrases.)
    2. Embed reference + all paraphrases.
    3. Compute cosine(ref, x) for every x.
    4. Set threshold so that good paraphrases land above it and bad ones below.

Threshold rule:
    threshold = max( min(good_sims),  max(bad_sims) + epsilon )

    - min(good_sims) is the floor of the "acceptable" distribution; anything
      below it is probably a worse-than-paraphrase answer.
    - max(bad_sims) + epsilon ensures bad examples are explicitly excluded
      even if the good floor happens to dip below them (which means the bad
      set was too plausible and the good set drifted — a calibration warning).

We also store mean/stddev of the good distribution into the note for the
user to inspect later, and a `_grader_calibration` JSON blob with the raw
numbers in case they want to audit. The threshold field is what the runtime
actually reads.
"""

from __future__ import annotations

import json
import math
from typing import Iterable

# aqt imports are guarded so this module is importable in headless tests.
# precompute_calibration() runs entirely without aqt; apply_calibration() and
# calibrate_selected_notes() require it and will raise at call time if missing.
try:
    from aqt import mw
    from aqt.qt import QProgressDialog, Qt
except ImportError:
    mw = None  # type: ignore[assignment]
    QProgressDialog = Qt = None  # type: ignore[assignment]

from .config import get_config
from .grader import cosine
from .openai_client import batch_embed, generate_keywords, generate_paraphrases


THRESHOLD_FIELD = "_grader_threshold"
DIAGNOSTICS_FIELD = "_grader_calibration"
REF_EMBEDDING_FIELD = "_grader_ref_embedding"
KEYWORDS_FIELD = "Keywords"
ANSWER_FIELD_CANDIDATES = ("Back", "Answer", "Definition")
QUESTION_FIELD_CANDIDATES = ("Front", "Question", "Prompt")


def _find_field(note, candidates) -> str | None:
    for name in candidates:
        if name in note:
            return name
    return None


def _find_answer_field(note) -> str | None:
    return _find_field(note, ANSWER_FIELD_CANDIDATES)


def _ensure_field(model, field_name: str) -> bool:
    """Add `field_name` to `model` if missing. Returns True if added."""
    existing = {f["name"] for f in model["flds"]}
    if field_name in existing:
        return False
    mm = mw.col.models
    fld = mm.new_field(field_name)
    mm.add_field(model, fld)
    mm.save(model)
    return True


def _summary_stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, stddev). stddev=0 for a single value."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def precompute_calibration(
    *,
    question: str,
    reference: str,
    keywords: list[str] | None = None,
) -> dict:
    """Run the LLM + embedding work for one card. **No Anki collection access.**

    Designed to be called from a background thread (e.g. book_to_cards's
    worker) so the Anki main thread isn't blocked for ~10s per card. The
    returned dict carries everything apply_calibration() needs to write the
    note fields on the main thread in a near-instant operation.

    If `keywords` is omitted, the model is asked to pick them. All embeddings
    (reference + good + bad) are fetched in a single batched HTTP call.
    """
    if not reference:
        raise ValueError("Reference text is empty.")
    cfg = get_config()
    if not keywords:
        keywords = generate_keywords(question, reference)
    paraphrases = generate_paraphrases(reference, keywords)
    good_texts = paraphrases["good"]
    bad_texts = paraphrases["bad"]

    # One HTTP call for every vector we'll need.
    all_texts = [reference] + good_texts + bad_texts
    all_vecs = batch_embed(all_texts)
    ref_vec = all_vecs[0]
    good_vecs = all_vecs[1:1 + len(good_texts)]
    bad_vecs = all_vecs[1 + len(good_texts):]

    good_sims = [cosine(ref_vec, v) for v in good_vecs]
    bad_sims = [cosine(ref_vec, v) for v in bad_vecs]
    good_mean, good_std = _summary_stats(good_sims)
    bad_mean, bad_std = _summary_stats(bad_sims)

    floor_of_good = min(good_sims) if good_sims else cfg["fallback_threshold"]
    ceiling_of_bad = max(bad_sims) if bad_sims else 0.0
    threshold = max(floor_of_good, ceiling_of_bad + cfg["epsilon_above_bad"])
    overlap = ceiling_of_bad >= floor_of_good

    return {
        "keywords": keywords,
        "threshold": threshold,
        "ref_vec": ref_vec,
        "diagnostics": {
            "threshold": threshold,
            "good_mean": good_mean,
            "good_std": good_std,
            "good_min": min(good_sims) if good_sims else None,
            "bad_mean": bad_mean,
            "bad_std": bad_std,
            "bad_max": max(bad_sims) if bad_sims else None,
            "overlap_warning": overlap,
            "n_good": len(good_sims),
            "n_bad": len(bad_sims),
        },
    }


def apply_calibration(note, calibration: dict) -> dict:
    """Write pre-computed calibration values onto a note. **Must run on
    Anki's main thread** (uses mw.col).

    Ensures the model has the required fields, then writes
    Keywords / _grader_threshold / _grader_ref_embedding / _grader_calibration.
    """
    model = note.note_type()
    added = False
    for fname in (KEYWORDS_FIELD, THRESHOLD_FIELD, DIAGNOSTICS_FIELD, REF_EMBEDDING_FIELD):
        if _ensure_field(model, fname):
            added = True
    if added:
        note = mw.col.get_note(note.id)

    # If keywords were pre-typed by the user, keep theirs. Otherwise stamp
    # the model-picked ones from precompute_calibration.
    existing_kw = note[KEYWORDS_FIELD].strip() if KEYWORDS_FIELD in note else ""
    if not existing_kw:
        note[KEYWORDS_FIELD] = ", ".join(calibration["keywords"])

    note[THRESHOLD_FIELD] = f"{calibration['threshold']:.4f}"
    note[REF_EMBEDDING_FIELD] = json.dumps(calibration["ref_vec"])
    note[DIAGNOSTICS_FIELD] = json.dumps(calibration["diagnostics"], indent=2)
    mw.col.update_note(note)
    return {"nid": note.id, "threshold": calibration["threshold"],
            "overlap": calibration["diagnostics"]["overlap_warning"]}


def _calibrate_one(note) -> dict:
    """End-to-end calibration of a single Anki note.

    Convenience wrapper that combines precompute_calibration (slow, no col)
    and apply_calibration (fast, needs col). Used by the smart_grader menu
    flow which runs entirely on the main thread.
    """
    answer_field = _find_answer_field(note)
    if answer_field is None:
        raise ValueError("No answer field found (looked for Back/Answer/Definition).")
    question_field = _find_field(note, QUESTION_FIELD_CANDIDATES)
    reference = note[answer_field].strip()
    question = note[question_field].strip() if question_field else ""

    existing_kw = note[KEYWORDS_FIELD].strip() if KEYWORDS_FIELD in note else ""
    keywords = [k.strip() for k in existing_kw.split(",") if k.strip()] or None

    calibration = precompute_calibration(
        question=question, reference=reference, keywords=keywords,
    )
    return apply_calibration(note, calibration)


def calibrate_selected_notes(nids: Iterable[int], col) -> tuple[int, int]:
    """
    Run calibration over the given note IDs with a progress dialog.
    Returns (n_calibrated, n_skipped).
    """
    nids = list(nids)
    progress = QProgressDialog("Calibrating cards…", "Cancel", 0, len(nids), mw)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)

    done = skipped = 0
    for i, nid in enumerate(nids):
        if progress.wasCanceled():
            break
        progress.setValue(i)
        progress.setLabelText(f"Calibrating {i+1}/{len(nids)}…")
        mw.app.processEvents()

        try:
            note = col.get_note(nid)
            _calibrate_one(note)
            done += 1
        except Exception as e:
            # Skip and continue — partial progress is better than an aborted batch.
            print(f"[smart_grader] skipped nid={nid}: {e}")
            skipped += 1
    progress.setValue(len(nids))
    return done, skipped


# Backwards-compatible alias. New code should import from smart_grader.api.
calibrate_note = _calibrate_one
