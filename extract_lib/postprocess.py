from __future__ import annotations

import re
from typing import Any, Dict, List

from extract_lib.schema_loader import SchemaField

_MAX_EVIDENCE_CHARS = 300

def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

def _normalize_value(v: str) -> str:
    # strip stray citation markers like [1], (2024), trailing punctuation
    v = re.sub(r"\s*\[[\d,\s]+\]\s*$", "", v)
    v = re.sub(r"\s*\(\d{4}[a-z]?\)\s*$", "", v)
    return v.strip(" .,;:")

def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        key = v.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out

def _coerce_values(payload: Any) -> List[str]:
    """Accept several legacy shapes and return a clean list of value strings."""
    if payload is None:
        return []
    if isinstance(payload, list):
        raw_vals = payload
    elif isinstance(payload, dict):
        # Prefer "values" if present and non-empty; otherwise fall back to
        # the singular "value" / "name" keys that older prompts produced.
        if "values" in payload and payload["values"]:
            raw_vals = payload["values"]
        else:
            v = payload.get("value") if "value" in payload else payload.get("name")
            if isinstance(v, list):
                raw_vals = v
            elif v not in (None, ""):
                raw_vals = [v]
            else:
                raw_vals = []
    else:
        raw_vals = [payload]

    cleaned: List[str] = []
    for item in raw_vals:
        if isinstance(item, dict):
            # e.g. {"value": "...", "evidence": "..."} -> take "value"
            s = _as_str(item.get("value") or item.get("name") or "")
        else:
            s = _as_str(item)
        s = _normalize_value(s)
        if s:
            cleaned.append(s)
    return _dedupe_preserve_order(cleaned)

def _coerce_evidence(payload: Any) -> List[Dict[str, str]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        raw_ev = payload.get("evidence", [])
    elif isinstance(payload, list):
        raw_ev = []
    else:
        raw_ev = []
    if isinstance(raw_ev, str):
        raw_ev = [raw_ev]
    elif isinstance(raw_ev, dict):
        raw_ev = [raw_ev]
    out: List[Dict[str, str]] = []
    for ev in raw_ev or []:
        if isinstance(ev, dict):
            quote = _as_str(ev.get("quote") or ev.get("snippet") or ev.get("text"))
            passage = _as_str(ev.get("passage") or ev.get("source") or "")
        else:
            quote = _as_str(ev)
            passage = ""
        if quote:
            if len(quote) > _MAX_EVIDENCE_CHARS:
                quote = quote[: _MAX_EVIDENCE_CHARS - 1].rstrip() + "…"
            out.append({"quote": quote, "passage": passage})
    return out

def normalize_extraction(
    raw_result: Dict[str, Any],
    fields: List[SchemaField],
) -> Dict[str, Any]:
    extractions_in = raw_result.get("extractions") or raw_result.get("extracted") or {}
    if not isinstance(extractions_in, dict):
        extractions_in = {}

    out_extractions: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        payload = extractions_in.get(f.name)
        values = _coerce_values(payload)
        # For single-valued fields (plain string), cap at one even if the LLM
        # over-produces. Array and enum fields keep all distinct values.
        if not f.is_multi_valued() and len(values) > 1:
            values = values[:1]
        evidence = _coerce_evidence(payload)
        out_extractions[f.name] = {"values": values, "evidence": evidence}

    return {
        "paper_id": raw_result.get("paper_id", ""),
        "extractions": out_extractions,
    }
