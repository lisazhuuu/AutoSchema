from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from extract_lib.config import (
    ALIAS_HIT_WEIGHT,
    EVIDENCE_HIT_WEIGHT,
    GLOBAL_PARAGRAPH_BUDGET,
    NAME_HIT_WEIGHT,
    PARAGRAPH_MAX_CHARS,
    PARAGRAPH_MIN_CHARS,
    TOP_K_PER_FIELD,
)
from extract_lib.schema_loader import SchemaField

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")

# Paragraph splitting
def split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    raw_paras = _PARA_SPLIT_RE.split(text)
    out: List[str] = []
    for p in raw_paras:
        p = p.strip()
        if len(p) < PARAGRAPH_MIN_CHARS:
            continue
        if len(p) <= PARAGRAPH_MAX_CHARS:
            out.append(p)
        else:
            # window long paragraph
            for i in range(0, len(p), PARAGRAPH_MAX_CHARS):
                window = p[i : i + PARAGRAPH_MAX_CHARS].strip()
                if len(window) >= PARAGRAPH_MIN_CHARS:
                    out.append(window)
    return out

# Scoring
def _tokenize(text: str) -> Set[str]:
    return set(t.lower() for t in _TOKEN_RE.findall(text or ""))

def _count_phrase_hits(haystack_lower: str, phrase: str) -> int:
    if not phrase:
        return 0
    if " " in phrase or "-" in phrase:
        return haystack_lower.count(phrase)
    return len(re.findall(rf"\b{re.escape(phrase)}\b", haystack_lower))

def score_paragraph_for_field(paragraph: str, field: SchemaField) -> float:
    if not paragraph:
        return 0.0
    p_lower = paragraph.lower()
    score = 0.0

    # Canonical name hit gets the biggest boost
    for variant in {field.name.lower(), field.name.replace("_", " ").lower()}:
        score += NAME_HIT_WEIGHT * _count_phrase_hits(p_lower, variant)

    # Aliases / member field names
    for alias in field.aliases + field.member_field_names:
        for variant in {alias.lower(), alias.replace("_", " ").lower()}:
            score += ALIAS_HIT_WEIGHT * _count_phrase_hits(p_lower, variant)

    # Evidence-snippet token overlap (mild)
    if field.evidence_snippets and score > 0:
        para_tokens = _tokenize(paragraph)
        for snippet in field.evidence_snippets[:8]:
            overlap = len(para_tokens & _tokenize(snippet))
            if overlap:
                score += EVIDENCE_HIT_WEIGHT * overlap

    return score

# Top-K selection
@dataclass
class RetrievedContext:
    # Mapping field_name -> ordered list of paragraph indices (most relevant first)
    per_field_indices: Dict[str, List[int]]
    # The full list of paragraphs (so callers can pull text by index)
    paragraphs: List[str]
    # The de-duplicated, globally-budgeted set of paragraph indices to send to LLM
    selected_indices: List[int]

def retrieve_per_field_contexts(
    full_text: str,
    fields: List[SchemaField],
    top_k_per_field: int = TOP_K_PER_FIELD,
    global_budget: int = GLOBAL_PARAGRAPH_BUDGET,
) -> RetrievedContext:
    paragraphs = split_paragraphs(full_text)

    per_field: Dict[str, List[int]] = {}
    selected: List[int] = []
    seen: Set[int] = set()

    # First pass: per-field top-K
    for f in fields:
        scored: List[Tuple[int, float]] = []
        for idx, para in enumerate(paragraphs):
            s = score_paragraph_for_field(para, f)
            if s > 0:
                scored.append((idx, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_idx = [idx for idx, _ in scored[:top_k_per_field]]
        per_field[f.name] = top_idx
        for idx in top_idx:
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)

    # Fallback: if nothing scored (rare), keep first few paragraphs so that the LLM always has some context
    if not selected and paragraphs:
        selected = list(range(min(len(paragraphs), 10)))

    # Apply global budget; keep relative order from the per-field walk
    if len(selected) > global_budget:
        selected = selected[:global_budget]

    return RetrievedContext(
        per_field_indices=per_field,
        paragraphs=paragraphs,
        selected_indices=selected,
    )

def render_context_pack(ctx: RetrievedContext) -> str:
    """Render the selected paragraphs as a single string for the LLM prompt."""
    lines: List[str] = []
    for i, idx in enumerate(ctx.selected_indices, 1):
        lines.append(f"[Passage {i}] {ctx.paragraphs[idx]}")
    return "\n\n".join(lines)

# Paper-level relevance signal (used for --top-n ranking)
@dataclass
class PaperRelevance:
    """How relevant a paper looks BEFORE we spend any LLM tokens on it."""
    n_fields_covered: int    # how many schema fields had at least one matching paragraph
    total_best_score: float  # sum of the single best paragraph score per field
    n_paragraphs: int        # total paragraphs in the paper (after split)

def paper_relevance_signal(
    full_text: str,
    fields: List[SchemaField],
) -> PaperRelevance:
    paragraphs = split_paragraphs(full_text)
    if not paragraphs:
        return PaperRelevance(n_fields_covered=0, total_best_score=0.0, n_paragraphs=0)

    total = 0.0
    n_covered = 0
    for f in fields:
        best = 0.0
        for p in paragraphs:
            s = score_paragraph_for_field(p, f)
            if s > best:
                best = s
        if best > 0:
            n_covered += 1
            total += best
    return PaperRelevance(
        n_fields_covered=n_covered,
        total_best_score=total,
        n_paragraphs=len(paragraphs),
    )
