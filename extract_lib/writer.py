from __future__ import annotations

import csv, json
from pathlib import Path
from typing import Dict, Iterable, List, Set

from extract_lib.config import MULTIVALUE_JOINER, NA_VALUE
from extract_lib.schema_loader import SchemaField

# JSONL append
def append_jsonl(jsonl_path: Path, record: Dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_done_paper_ids(jsonl_path: Path) -> Set[str]:
    """Resume support: return paper_ids already written to the jsonl."""
    if not jsonl_path.exists():
        return set()
    done: Set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("paper_id")
            if pid:
                done.add(pid)
    return done

# CSV rewrite
def _format_cell(values: List[str]) -> str:
    if not values:
        return NA_VALUE
    return MULTIVALUE_JOINER.join(values)

def rewrite_csv(
    csv_path: Path,
    fields: List[SchemaField],
    records: Iterable[Dict],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [f.name for f in fields]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_id"] + field_names)
        writer.writeheader()
        for rec in records:
            row = {"paper_id": rec.get("paper_id", "")}
            ext = rec.get("extractions", {}) or {}
            for name in field_names:
                payload = ext.get(name, {}) or {}
                row[name] = _format_cell(payload.get("values", []) or [])
            writer.writerow(row)

def load_all_records(jsonl_path: Path) -> List[Dict]:
    if not jsonl_path.exists():
        return []
    out: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out