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

from aqt import mw
from aqt.qt import QProgressDialog, Qt

from .config import get_config
from .grader import cosine
from .openai_client import embed, generate_keywords, generate_paraphrases


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


def calibrate_note(note) -> dict:
    """
    Calibrate a single note. Returns a diagnostics dict; raises on hard failures
    (missing fields, API errors) so the caller can decide whether to skip or abort.
    """
    cfg = get_config()
    answer_field = _find_answer_field(note)
    if answer_field is None:
        raise ValueError("No answer field found (looked for Back/Answer/Definition).")
    question_field = _find_field(note, QUESTION_FIELD_CANDIDATES)

    reference = note[answer_field].strip()
    if not reference:
        raise ValueError("Answer field is empty.")
    question = note[question_field].strip() if question_field else ""

    # Make sure target fields exist on this note type before we try to write.
    # Keywords is included so calibration works on note types that never had it.
    model = note.note_type()
    added_kw = _ensure_field(model, KEYWORDS_FIELD)
    added_threshold = _ensure_field(model, THRESHOLD_FIELD)
    added_diag = _ensure_field(model, DIAGNOSTICS_FIELD)
    added_ref = _ensure_field(model, REF_EMBEDDING_FIELD)
    if added_kw or added_threshold or added_diag or added_ref:
        # Re-load the note so it picks up the new fields.
        note = mw.col.get_note(note.id)

    # Keywords are AI-generated when the field is empty so the user only needs
    # to author the question + answer. User-typed keywords are left alone.
    existing_kw = note[KEYWORDS_FIELD].strip() if KEYWORDS_FIELD in note else ""
    if not existing_kw:
        keywords = generate_keywords(question, reference)
        note[KEYWORDS_FIELD] = ", ".join(keywords)
    else:
        keywords = [k.strip() for k in existing_kw.split(",") if k.strip()]

    paraphrases = generate_paraphrases(reference, keywords)
    good_texts = paraphrases["good"]
    bad_texts = paraphrases["bad"]

    ref_vec = embed(reference)
    good_sims = [cosine(ref_vec, embed(t)) for t in good_texts]
    bad_sims = [cosine(ref_vec, embed(t)) for t in bad_texts]

    good_mean, good_std = _summary_stats(good_sims)
    bad_mean, bad_std = _summary_stats(bad_sims)

    # Core threshold rule. Note this is intentionally NOT "mean - 2*std" — that
    # framing assumes a Gaussian, but with ~12 samples and bounded support in
    # [-1, 1], the floor-of-good / ceiling-of-bad rule is more robust.
    floor_of_good = min(good_sims) if good_sims else cfg["fallback_threshold"]
    ceiling_of_bad = max(bad_sims) if bad_sims else 0.0
    threshold = max(floor_of_good, ceiling_of_bad + cfg["epsilon_above_bad"])

    # If the bad ceiling is above the good floor, the LLM gave us overlapping
    # distributions — calibration is unreliable for this card. We still pick
    # the more conservative threshold, but flag it.
    overlap = ceiling_of_bad >= floor_of_good
    note[THRESHOLD_FIELD] = f"{threshold:.4f}"
    # Persist the reference embedding so review never has to recompute it.
    note[REF_EMBEDDING_FIELD] = json.dumps(ref_vec)
    note[DIAGNOSTICS_FIELD] = json.dumps({
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
    }, indent=2)
    mw.col.update_note(note)
    return {"nid": note.id, "threshold": threshold, "overlap": overlap}


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
            calibrate_note(note)
            done += 1
        except Exception as e:
            # Skip and continue — partial progress is better than an aborted batch.
            print(f"[smart_grader] skipped nid={nid}: {e}")
            skipped += 1
    progress.setValue(len(nids))
    return done, skipped
