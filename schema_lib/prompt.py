from __future__ import annotations
from typing import Any, Dict, List
from .domain import SchemaDomainSpec

def build_extraction_prompt(
    final_fields: List[Dict[str, Any]],
    spec: SchemaDomainSpec,
    last_tier_text: str,
) -> str:
    if not final_fields:
        raise SystemExit("Cannot build extraction prompt: final schema is empty.")

    name_list = [f["name"] for f in final_fields]

    field_block_lines: List[str] = []
    for f in final_fields:
        line = f"- {f['name']}: {f['description']}"
        examples = (f.get("examples") or [])[:2]
        if examples:
            line += f"  (examples: {', '.join(map(str, examples))})"
        field_block_lines.append(line)
    field_block = "\n".join(field_block_lines)

    extracted_skeleton = ",\n      ".join(
        f'"{n}": "<value or N/A>"' for n in name_list
    )
    evidence_skeleton = ",\n      ".join(
        f'"{n}": "<supporting verbatim quote or NO_EVIDENCE_FOUND>"' for n in name_list
    )

    return f"""
        [ROLE]
        You are a {spec.extractor_role}.

        [TASK]
        Extract structured information about "{last_tier_text}" from the paper text
        into the schema below. For every field, also provide a verbatim quote from
        the paper that supports the extracted value (evidence-grounded extraction).

        [SCHEMA]
        {field_block}

        [OUTPUT FORMAT]
        Return ONLY valid JSON of this exact shape (no Markdown, no commentary):
        {{
        "extracted": {{
            {extracted_skeleton}
        }},
        "evidence": {{
            {evidence_skeleton}
        }}
        }}

        [RULES]
        1. Preserve numerical values and units exactly when present in the text.
        2. Match each value to the correct entity / condition / cohort mentioned in the text.
        3. If a value is missing from the paper, write "N/A" in "extracted" AND
        "NO_EVIDENCE_FOUND" in "evidence" for that field.
        4. Each evidence quote MUST be copied verbatim from the paper and be <= 40 words.
        5. Do not invent fields outside the schema. Do not output any field
        that is not listed in [SCHEMA].
        6. The keys in "extracted" and "evidence" MUST exactly match the field names
        in [SCHEMA].
    """.strip()