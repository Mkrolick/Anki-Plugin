"""Checkpoint: atomic writes + resume rules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_atomic_write_replaces_existing(tmp_path):
    from book_to_cards.pipeline.checkpoint import write_json_atomic
    p = tmp_path / "x.json"
    p.write_text('{"old": true}')
    write_json_atomic(p, {"new": True})
    assert json.loads(p.read_text()) == {"new": True}
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
    from book_to_cards.pipeline.checkpoint import detect_resume_state, write_json_atomic, append_jsonl, mark_complete
    write_json_atomic(tmp_path / "extract.json", {"book_hash": "abc", "pages": []})
    append_jsonl(tmp_path / "map.jsonl", {"chunk_index": 0, "aspects": []})
    mark_complete(tmp_path / "map.jsonl")
    state = detect_resume_state(tmp_path)
    assert state.next_stage == "merge"


def test_manifest_mismatch_triggers_cold_restart_flag(tmp_path):
    from book_to_cards.pipeline.checkpoint import detect_resume_state, write_json_atomic
    write_json_atomic(tmp_path / "manifest.json", {"book_hash": "abc", "chunk_size": 5, "overlap": 1})
    write_json_atomic(tmp_path / "extract.json", {"book_hash": "abc", "pages": []})
    state = detect_resume_state(tmp_path, expected_manifest={"book_hash": "abc", "chunk_size": 10, "overlap": 1})
    assert state.cold_restart is True
