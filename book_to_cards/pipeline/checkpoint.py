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
    map_resume_from: int = 0
    reduce_resume_from: int = 0
    cold_restart: bool = False


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
    return sum(1 for r in read_jsonl(path) if r.get("_complete") is not True)


def mark_complete(jsonl_path: Path) -> None:
    append_jsonl(jsonl_path, {"_complete": True})


def detect_resume_state(run_dir: Path, *, expected_manifest: dict | None = None) -> ResumeState:
    run_dir = Path(run_dir)

    manifest_path = run_dir / "manifest.json"
    if expected_manifest is not None and manifest_path.exists():
        actual = json.loads(manifest_path.read_text())
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
