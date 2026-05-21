import json, re
from pathlib import Path
from typing import List, Dict

# Normalize title for deduplication
def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())

# Load and normalize queries.json
def load_queries(path: str) -> List[Dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("queries", [])
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported queries.json format: {type(raw)}")

    fixed: List[Dict] = []
    for q in items:
        if isinstance(q, str):
            fixed.append({"query_bool": q, "query_nl": q, "why": "", "tiers": []})
            continue
        if not isinstance(q, dict):
            continue
        qb = q.get("query_bool") or q.get("query_nl") or q.get("query") or ""
        qn = q.get("query_nl")   or qb
        fixed.append({
            "query_bool": qb,
            "query_nl":   qn,
            "why":        q.get("why", ""),
            "tiers":      q.get("tiers", []) or [],
        })
    return fixed
