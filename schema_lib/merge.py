from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .domain import SchemaDomainSpec

_MERGE_SYSTEM = "Return ONLY valid JSON. No prose, no markdown, no commentary."

def _build_merge_prompt(
    final_fields: List[Dict[str, Any]],
    spec: SchemaDomainSpec,
    last_tier_text: str,
) -> str:
    field_lines = "\n".join(
        f"  - {f.get('name')}: {(f.get('description') or '').strip()}"
        for f in final_fields
    )
    domain_rules = getattr(spec, "merge_rules", None) or [
        "Keep only one field per underlying concept.",
    ]
    domain_rule_block = "\n              - ".join(domain_rules)

    return f"""
        You are a {spec.extractor_role} performing a FINAL schema cleanup.

        Subdomain focus: "{last_tier_text}"

        Below is a candidate schema (a list of fields). Remove semantic
        redundancy WITHOUT inventing new fields.

        GENERIC RULES:
            - Merge fields that capture the SAME underlying concept into ONE.
            - When several fields are variants/measurements of the same
              quantity, keep ONE of them.
            - Drop a broad / generic field if one or more MORE SPECIFIC fields
              already cover its information.
            - You may only reuse names that already appear below. Never invent
              a new field name. Never add a brand-new field.

        DOMAIN-SPECIFIC RULES:
              - {domain_rule_block}

        CURRENT FIELDS:
{field_lines}

        Return ONLY valid JSON of this exact shape:
        {{
          "groups": [
            {{
              "keep": "<an existing field name to keep>",
              "merge_from": ["<existing field name>", "..."],
              "description": "<one-sentence merged description>",
              "reason": "<short why these are the same concept>"
            }}
          ],
          "drop": [
            {{"name": "<an existing field name to drop>", "reason": "<short why>"}}
          ]
        }}

        JSON rules:
            - "keep", every "merge_from" entry, and every "drop" name MUST be a
              field name that appears in CURRENT FIELDS verbatim.
            - A field name may appear in AT MOST one group, and not also in drop.
            - Fields you do not mention are kept unchanged.
            - If nothing needs merging or dropping, return
              {{"groups": [], "drop": []}}.
    """.strip()

def _merge_records(keep_name: str, member_names: List[str],
                   by_name: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(by_name[keep_name])
    examples = list(base.get("examples") or [])
    evidence = list(base.get("evidence_examples") or [])
    support = int(base.get("support_count") or 0)

    folded: List[str] = []
    for n in member_names:
        if n == keep_name or n not in by_name:
            continue
        r = by_name[n]
        for ex in (r.get("examples") or []):
            if ex and ex not in examples:
                examples.append(ex)
        for ev in (r.get("evidence_examples") or []):
            if ev and ev not in evidence:
                evidence.append(ev)
        support = max(support, int(r.get("support_count") or 0))
        folded.append(n)

    base["examples"] = examples[:5]
    base["evidence_examples"] = evidence[:3]
    base["support_count"] = support
    if folded:
        base["merged_from"] = folded
        base["source"] = "merged"
    return base

def _apply_merge_plan(
    final_fields: List[Dict[str, Any]],
    plan: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_name = {f["name"]: f for f in final_fields}
    order = [f["name"] for f in final_fields]

    groups_in = plan.get("groups") or []
    drops_in = plan.get("drop") or []

    drop_names = {(d.get("name") or "").strip()
                  for d in drops_in if isinstance(d, dict)}
    drop_names = {n for n in drop_names if n in by_name}

    consumed: set = set()                  # merge_from members folded away
    keep_repr: Dict[str, Dict[str, Any]] = {}
    applied_groups: List[Dict[str, Any]] = []

    for g in groups_in:
        if not isinstance(g, dict):
            continue
        keep = (g.get("keep") or "").strip()
        if keep not in by_name or keep in consumed or keep in keep_repr:
            continue
        mfrom = [n.strip() for n in (g.get("merge_from") or [])
                 if isinstance(n, str)]
        mfrom = [n for n in mfrom
                 if n in by_name and n != keep
                 and n not in consumed and n not in keep_repr]
        merged = _merge_records(keep, [keep] + mfrom, by_name)
        desc = (g.get("description") or "").strip()
        if desc:
            merged["description"] = desc
        keep_repr[keep] = merged
        for n in mfrom:
            consumed.add(n)
        drop_names.discard(keep)           # never drop a kept field
        if mfrom:
            applied_groups.append({"keep": keep, "merge_from": mfrom,
                                   "reason": (g.get("reason") or "").strip()})

    # Build output in original order.
    out: List[Dict[str, Any]] = []
    emitted: set = set()
    for name in order:
        if name in consumed:
            continue
        if name in drop_names and name not in keep_repr:
            continue
        if name in keep_repr:
            if name not in emitted:
                out.append(keep_repr[name])
                emitted.add(name)
            continue
        out.append(by_name[name])

    summary = {
        "groups_applied": applied_groups,
        "dropped": sorted(n for n in drop_names if n not in keep_repr),
    }
    return out, summary

def semantic_merge_fields(
    final_fields: List[Dict[str, Any]],
    spec: SchemaDomainSpec,
    last_tier_text: str,
    max_final_fields: int,
    llm_chat_fn=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "applied": False,
        "before": len(final_fields),
        "after": len(final_fields),
    }
    if not final_fields or len(final_fields) < 2:
        debug["reason"] = "fewer than 2 fields; nothing to merge"
        return final_fields, debug

    if llm_chat_fn is None:
        # Imported lazily so this module stays importable without credentials.
        from prompt_lib.llm_client import llm_chat as llm_chat_fn  # type: ignore

    prompt = _build_merge_prompt(final_fields, spec, last_tier_text)
    try:
        raw = llm_chat_fn(
            [
                {"role": "system", "content": _MERGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            force_json=True,
        )
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            raise ValueError("merge plan is not a JSON object")
    except Exception as e:
        debug["reason"] = f"LLM merge failed, kept un-merged schema: {e}"
        print(f"[schema][merge] WARNING: {debug['reason']}")
        return final_fields, debug

    merged, summary = _apply_merge_plan(final_fields, plan)

    # Safety: never return an empty schema; never grow the field count.
    if not merged:
        debug["reason"] = "merge produced an empty schema; kept un-merged"
        print(f"[schema][merge] WARNING: {debug['reason']}")
        return final_fields, debug
    if len(merged) > max_final_fields:
        merged = merged[:max_final_fields]

    debug.update({
        "applied": True,
        "after": len(merged),
        "groups_applied": summary["groups_applied"],
        "dropped": summary["dropped"],
    })
    return merged, debug