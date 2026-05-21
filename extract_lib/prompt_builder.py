from __future__ import annotations

from typing import List

from extract_lib.schema_loader import SchemaField

# System prompt
_DOMAIN_INTRO = (
    "You are a careful scientific information-extraction assistant. You read "
    "research papers and extract values for a fixed set of fields. The field "
    "descriptions, aliases, and example evidence listed below define the "
    "relevant scientific domain — use them, together with the passages you "
    "are given, to ground your understanding of the terminology you will "
    "encounter."
)

_GENERAL_RULES = """[EXTRACTION RULES]
1. EVIDENCE FIRST. Only extract values that are directly supported by the
   passages provided. If the passages do not state the value, return an empty
   list for that field — never invent or hallucinate a value.

2. MULTI-VALUE. Fields with value_type "array" or "enum" may carry multiple
   distinct values. Return EVERY distinct value the paper discusses.
   Fields with value_type "string" hold a single value — return a one-element
   list (kept inside a list for output consistency).

3. SPLIT COMPOUND PHRASES. If the paper writes a list joined by "and" / "or"
   / "/" — e.g. "type A and type B", "X-, Y-, and Z- enzymes", or "A / B" —
   return the items as SEPARATE values:
       "type A and type B"           ->  ["type A", "type B"]
       "X-, Y-, and Z- enzymes"      ->  ["X-enzyme", "Y-enzyme", "Z-enzyme"]
       "A / B"                       ->  ["A", "B"]
   Never return a single string that contains "and" / "or" / "/" joining two
   distinct values.

4. VALUES ARE SHORT, NORMALIZED TERMS — NOT SENTENCES. A value is the answer,
   not the supporting clause. Strip determiners, verbs, citations, and
   surrounding prose; keep only the canonical term or phrase.
     BAD  "the level of X was elevated due to decreased clearance"
     GOOD "elevated"
     BAD  "absolute amount of X increased significantly under condition Y"
     GOOD "increased"
     BAD  "X was normalized against Y as a loading control"
                                       (← methodology, not a value of X)
     GOOD ""   (leave the values list empty for this field)
   For direction / level / ratio / state fields, prefer a concise canonical
   token when the paper supports it — e.g. "increased", "decreased",
   "elevated", "reduced", "unchanged", "no change". If the paper gives a
   numeric change, use a short form such as "↑ 2-fold" or "34-fold increase".

5. SYNONYM AWARENESS. A passage may refer to a field by an alias rather than
   its canonical name. The alias list under each field tells you which terms
   to treat as equivalent.

6. VALUE ≠ EVIDENCE. The "values" array holds the answer terms. The
   "evidence" array holds the verbatim snippet that supports each value.
   NEVER copy a full evidence sentence into "values".

7. EVIDENCE QUOTES. For every value you return, include a short verbatim
   snippet (≤ 30 words) from the passages, plus the passage index
   (e.g. "Passage 4"). values and evidence should be the same length and
   align element-by-element.

8. DO NOT GUESS. Returning an empty list is strictly better than fabricating.

9. NORMALIZE WORDING. Use the natural wording the paper uses, but strip
   trailing punctuation, citation numbers (e.g. [12]), reference year
   markers (e.g. "(2019)"), and reference IDs.
"""

_OUTPUT_FORMAT = """[OUTPUT FORMAT — return a single JSON object]
{
  "paper_id": "<string>",
  "extractions": {
    "<field_name>": {
      "values":   [<short term>, ...],      // empty list = N/A
      "evidence": [{"quote": "<≤30-word snippet>", "passage": "Passage N"}, ...]
    },
    ...
  }
}
The keys in "extractions" MUST be exactly the field names listed below — no
extras, no omissions. Length of "values" and "evidence" should match. Do not
return anything outside this JSON object.
"""

def _render_field_block(f: SchemaField) -> str:
    """One field's spec block: name, type, description, aliases, examples."""
    lines: List[str] = []
    lines.append(f"- {f.name}  (value_type: {f.value_type})")
    if f.description:
        lines.append(f"    description: {f.description}")
    if f.aliases:
        # cap so the prompt doesn't balloon — the most useful aliases come first
        shown = list(dict.fromkeys(f.aliases))[:10]
        lines.append("    aliases: " + ", ".join(shown))
    if f.evidence_snippets:
        examples = [s.strip() for s in f.evidence_snippets if s.strip()][:2]
        if examples:
            ex_lines = "; ".join(f'"{e}"' for e in examples)
            lines.append(f"    example evidence: {ex_lines}")
    return "\n".join(lines)

def build_system_prompt(fields: List[SchemaField]) -> str:
    parts = [_DOMAIN_INTRO, "", _GENERAL_RULES.strip(), "", _OUTPUT_FORMAT.strip(), "", "[FIELDS]"]
    for f in fields:
        parts.append(_render_field_block(f))
        parts.append("")
    return "\n".join(parts).strip()

# User prompt
def build_user_prompt(paper_id: str, context_pack: str) -> str:
    return (
        f"paper_id: {paper_id}\n\n"
        f"Below are the passages from this paper that mention any of the "
        f"target fields or their aliases. Use only these passages as evidence.\n\n"
        f"{context_pack}"
    )
