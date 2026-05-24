"""Background runner that drives the four-stage pipeline.

Runs as a QThread so the Anki main thread stays responsive. Any work that
touches mw.col goes through mw.taskman.run_on_main(...) so the collection
isn't accessed concurrently with review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
from .pipeline.map_chunks import chunk_iter, run_map
from .pipeline.merge_tags import merge_topics
from .pipeline.reduce_cards import reduce_topic
from .pipeline.types import Aspect, Card, Page, Topic


def _hash_pdf(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _user_files_dir() -> Path:
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
            state = detect_resume_state(run_dir, expected_manifest=manifest)
        if state.next_stage == "calibrate":
            self._do_precalibrate(run_dir, client, resume_from=state.calibrate_resume_from)

        summary = self._do_insert(run_dir)
        summary["cost_usd"] = client.meter.usd
        self.finished_with_summary.emit(summary)

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
        pages = [Page(index=p["index"], text=p["text"]) for p in data["pages"]]

        cfg = get_config()
        total = sum(1 for _ in chunk_iter(pages, size=cfg["chunk_pages"], overlap=cfg["chunk_overlap"]))

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

        # Load the page-text map once so reduce_topic can include surrounding
        # context, not just the verbatim quotes. Pages whose lookup misses
        # (rare; some source_pages come through as strings) are silently
        # skipped — the quotes still anchor evidence.
        extract = json.loads((run_dir / "extract.json").read_text())
        pages_map: dict[int, str] = {}
        for p in extract.get("pages", []):
            try:
                pages_map[int(p["index"])] = p.get("text", "")
            except (KeyError, TypeError, ValueError):
                continue

        for i, topic in enumerate(topics):
            if i < resume_from:
                continue
            try:
                cards = reduce_topic(
                    topic,
                    llm=client,
                    model=get_config()["chat_model"],
                    pages=pages_map,
                )
                for c in cards:
                    append_jsonl(run_dir / "cards.jsonl", c.to_dict())
            except Exception as e:
                append_jsonl(run_dir / "errors.jsonl", {
                    "stage": "reduce", "topic": topic.canonical, "error": str(e),
                })
            self.topic_done.emit(i + 1, len(topics))
            self.cost_updated.emit(client.meter.usd)
            if client.meter.usd > self.max_cost_usd:
                raise RuntimeError(f"cost ceiling ${self.max_cost_usd} exceeded")
            if self._stop:
                raise RuntimeError("stopped by user")
        mark_complete(run_dir / "cards.jsonl")

    # Estimated marginal cost per card for the precalibrate stage. smart_grader's
    # generators run on its own OpenAI client (its own cost meter that the
    # book_to_cards dialog can't see), so we display this rough estimate so the
    # meter doesn't visibly freeze. ~$0.001 per card at gpt-4o-mini + Voyage
    # rates is the back-of-envelope figure from the design spec.
    _PER_CARD_CALIBRATE_USD = 0.001

    def _do_precalibrate(self, run_dir: Path, client: OpenAIClient, *, resume_from: int):
        """Background-thread stage: pre-compute calibration data for every card.

        Runs N workers in parallel (config: `calibrate_concurrency`). Each
        worker handles one card end-to-end: keywords → paraphrases → batched
        embeddings → threshold. Results are appended to `calibrated.jsonl`
        in completion order; each line carries the card's input index so
        resume can detect which inputs are still missing.

        Anki's main thread is never touched here.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.stage_started.emit("calibrate")
        self._calibrate_baseline_usd = float(client.meter.usd)
        records = [c for c in read_jsonl(run_dir / "cards.jsonl") if c.get("_complete") is not True]
        total = len(records)

        # Resume: skip any input index already present in calibrated.jsonl.
        # We treat presence of `_input_index` as the authoritative key so a
        # previous serial run (without the index) still has its rows usable —
        # but new work always tags its index.
        done_indices: set[int] = set()
        cal_path = run_dir / "calibrated.jsonl"
        if cal_path.exists():
            for rec in read_jsonl(cal_path):
                if rec.get("_complete") is True:
                    continue
                idx = rec.get("_input_index")
                if isinstance(idx, int):
                    done_indices.add(idx)
        # Legacy compatibility: resume_from is a count-based fallback when
        # _input_index isn't present (pre-parallel runs).
        if not done_indices and resume_from:
            done_indices = set(range(resume_from))

        if self.skip_calibration:
            for i, c in enumerate(records):
                if i in done_indices:
                    continue
                payload = {**c, "_input_index": i, "calibration": None}
                append_jsonl(cal_path, payload)
            mark_complete(cal_path)
            return

        try:
            from smart_grader.api import is_available, precompute_calibration
            sg_available = is_available()
        except ImportError:
            sg_available = False
            precompute_calibration = None

        if not sg_available or precompute_calibration is None:
            for i, c in enumerate(records):
                if i in done_indices:
                    continue
                payload = {**c, "_input_index": i, "calibration": None}
                append_jsonl(cal_path, payload)
            mark_complete(cal_path)
            return

        cfg = get_config()
        concurrency = max(1, int(cfg.get("calibrate_concurrency", 8)))

        # append_jsonl is not atomic for lines >PIPE_BUF (4KB). Our records
        # carry a 1536-float embedding, well over that. Serialise writes.
        write_lock = threading.Lock()
        completed = [len(done_indices)]  # mutable counter for the closure

        def _worker(i: int, c: dict) -> dict | None:
            if self._stop:
                return None
            try:
                cal = precompute_calibration(
                    question=c.get("front", ""),
                    reference=c.get("back", ""),
                )
                payload = {**c, "_input_index": i, "calibration": cal}
            except Exception as e:
                with write_lock:
                    append_jsonl(run_dir / "errors.jsonl", {
                        "stage": "calibrate",
                        "card_front": c.get("front", "")[:80],
                        "error": str(e),
                    })
                payload = {**c, "_input_index": i, "calibration": None}
            with write_lock:
                append_jsonl(cal_path, payload)
                completed[0] += 1
                done = completed[0]
            self.topic_done.emit(done, total)
            self.cost_updated.emit(
                self._calibrate_baseline_usd + done * self._PER_CARD_CALIBRATE_USD
            )
            return payload

        pending = [(i, c) for i, c in enumerate(records) if i not in done_indices]
        if not pending:
            mark_complete(cal_path)
            return

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_worker, i, c) for i, c in pending]
            for fut in as_completed(futures):
                if self._stop:
                    # Best-effort cancel; in-flight workers still finish.
                    for f in futures:
                        f.cancel()
                    raise RuntimeError("stopped by user")
                # Propagate worker exceptions (they shouldn't raise because
                # _worker swallows them, but be defensive).
                fut.result()

        mark_complete(cal_path)

    def _do_insert(self, run_dir: Path) -> dict:
        """Main-thread stage: write notes + apply pre-computed calibration.

        Reads from calibrated.jsonl (where _do_precalibrate left cards with
        their calibration data attached), then for each card creates the note
        and stamps the calibration fields. No LLM work happens here.
        """
        self.stage_started.emit("insert")
        calibrated_path = run_dir / "calibrated.jsonl"
        if calibrated_path.exists():
            source = calibrated_path
        else:
            source = run_dir / "cards.jsonl"
        records = [c for c in read_jsonl(source) if c.get("_complete") is not True]
        total = len(records)
        result_holder: dict = {"inserted": 0, "errors": 0}

        try:
            from smart_grader.api import apply_calibration as _apply
        except ImportError:
            _apply = None

        runner = self  # capture for the closure

        def _do_inserts():
            note_type = ensure_note_type(mw.col)
            did = ensure_deck(mw.col, runner.deck_name)
            for idx, c in enumerate(records):
                try:
                    card = Card(
                        front=c["front"],
                        back=c["back"],
                        topic=c["topic"],
                        source_pages=c.get("source_pages") or [],
                        source_quotes=c.get("source_quotes") or [],
                    )
                    nid = insert_card(
                        mw.col, card,
                        note_type=note_type, deck_id=did,
                        calibrate=False,  # apply pre-computed instead
                    )
                    cal = c.get("calibration")
                    if cal and _apply is not None:
                        note = mw.col.get_note(nid)
                        _apply(note, cal)
                    result_holder["inserted"] += 1
                except Exception:
                    result_holder["errors"] += 1
                if (idx + 1) % 5 == 0 or idx + 1 == total:
                    runner.topic_done.emit(idx + 1, total)
            mw.col.save()

        mw.taskman.run_on_main(_do_inserts)
        return result_holder
