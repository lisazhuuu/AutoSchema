from __future__ import annotations

from typing import Any, Dict, List, Tuple
from .domain import SchemaDomainSpec
from .fields import (
    _score_signals,
    canonicalize_field_name,
    _last_tier_tokens,
    _matches_avoid,
    _is_generic_name,
    _last_tier_relevant,
    _evidence_quality,
)

def _signals_for(name: str, description: str, evidence_quotes: List[str],
                 papers: List[str], spec: SchemaDomainSpec,
                 last_tier_text: str) -> Dict[str, Any]:
    tier_tokens = _last_tier_tokens(last_tier_text)
    return {
        "evidence_quality":   _evidence_quality(evidence_quotes),
        "last_tier_relevant": _last_tier_relevant(name, description, tier_tokens),
        "is_generic_name":    _is_generic_name(name, spec.generic_field_names),
        "matches_avoid":      _matches_avoid(name, description, spec.avoid_fields),
        "papers":             papers,
    }

def refine_schema(
    *,
    previous_fields: List[Dict[str, Any]],
    canonicals: List[Dict[str, Any]],
    spec: SchemaDomainSpec,
    last_tier_text: str,
    max_final_fields: int,
    min_support: int,
    no_new_support_penalty: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Index canonicals by canonical name + every name_variant.
    cans_by_key: Dict[str, Dict[str, Any]] = {}
    for c in canonicals:
        cans_by_key[c["name"]] = c
        for v in c.get("name_variants", []):
            cans_by_key.setdefault(v, c)

    debug: List[Dict[str, Any]] = []
    scored_pool: List[Tuple[float, Dict[str, Any], str]] = []
    consumed_canonicals = set()

    # ---- (a)+(b)+(c) iterate previous fields ----
    for prev in previous_fields:
        prev_name = canonicalize_field_name(prev.get("name", ""))
        if not prev_name:
            continue

        # Try to match a new canonical by name.
        match = cans_by_key.get(prev_name)
        prev_examples  = list(prev.get("examples") or [])
        prev_evidence  = list(prev.get("evidence_examples") or [])
        prev_support   = int(prev.get("support_count") or 0)
        description    = (prev.get("description") or "").strip()

        if match is not None:
            consumed_canonicals.add(match["name"])
            # Merge examples, evidence, papers.
            merged_examples = list(prev_examples)
            for ex in match.get("examples", []):
                if ex and ex not in merged_examples:
                    merged_examples.append(ex)

            merged_evidence = list(prev_evidence)
            for ev in match.get("evidence_examples", []):
                if ev and ev not in merged_evidence:
                    merged_evidence.append(ev)

            # Take the more informative description (prefer the longer one).
            new_desc = (match.get("description") or "").strip()
            if len(new_desc) > len(description):
                description = new_desc

            merged_papers = list((match["_signals"] or {}).get("papers", []))
            support_count = prev_support + match.get("support_count", 0)
            source = "refined"
            no_new_support = False
        else:
            merged_examples = prev_examples
            merged_evidence = prev_evidence
            merged_papers   = []
            support_count   = prev_support
            source = "previous"
            no_new_support = True

        signals = _signals_for(
            name=prev_name, description=description,
            evidence_quotes=merged_evidence, papers=merged_papers,
            spec=spec, last_tier_text=last_tier_text,
        )
        score = _score_signals(signals, support_count)
        if no_new_support:
            score -= no_new_support_penalty

        drop_reasons: List[str] = []
        if signals["matches_avoid"]:
            drop_reasons.append("matches_avoid_fields")
        if signals["is_generic_name"]:
            drop_reasons.append("generic_field_name")

        debug.append({
            "name":          prev_name,
            "description":   description,
            "support_count": support_count,
            "score":         round(score, 3),
            "signals":       signals,
            "drop_reasons":  drop_reasons,
            "source":        source,
        })

        if drop_reasons:
            continue

        scored_pool.append((score, {
            "name":              prev_name,
            "description":       description,
            "examples":          merged_examples[:5],
            "support_count":     support_count,
            "evidence_examples": merged_evidence[:3],
            "source":            source,
        }, source))

    # ---- (e) brand-new canonicals not seen in previous ----
    for c in canonicals:
        if c["name"] in consumed_canonicals:
            continue
        sig = c["_signals"]
        score = _score_signals(sig, c["support_count"])
        drop_reasons: List[str] = []
        if c["support_count"] < min_support:
            drop_reasons.append(f"support_count<{min_support}")
        if sig["matches_avoid"]:
            drop_reasons.append("matches_avoid_fields")
        if sig["is_generic_name"]:
            drop_reasons.append("generic_field_name")

        debug.append({
            "name":          c["name"],
            "description":   c["description"],
            "support_count": c["support_count"],
            "score":         round(score, 3),
            "signals":       sig,
            "drop_reasons":  drop_reasons,
            "source":        "new",
        })
        if drop_reasons:
            continue

        scored_pool.append((score, {
            "name":              c["name"],
            "description":       c["description"],
            "examples":          c["examples"],
            "support_count":     c["support_count"],
            "evidence_examples": c["evidence_examples"],
            "source":            "new",
        }, "new"))

    # ---- (f) final selection ----
    scored_pool.sort(key=lambda kv: kv[0], reverse=True)
    final_fields = [entry for _score, entry, _src in scored_pool[:max_final_fields]]

    return final_fields, debug
