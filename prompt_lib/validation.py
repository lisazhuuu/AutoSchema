from typing import Iterable

# ========== Some helper functions ==========
def _haystack(item: dict, qb_final: str) -> str:
    return " ".join([
        qb_final or "",
        item.get("query_nl") or "",
        " ".join(item.get("synonyms") or []),
    ]).lower()

def _norm(seq: Iterable[str]) -> list:
    return [str(s).strip().lower() for s in seq]

# Main validation function for query items, based on tier requirements and block terms.
def validate_tiers(item: dict, qb_final: str, DOMAIN: str, SPEC=None) -> bool:
    tiers = item.get("tiers") or []
    required_tiers = getattr(SPEC, "tiers", None) if SPEC is not None else None

    # 1. Exact tier-list match (case-insensitive, stripped).
    if required_tiers:
        if _norm(tiers) != _norm(required_tiers):
            return False

    # 2. Last-tier keyword / alias / required-term presence.
    if required_tiers:
        last_tier = required_tiers[-1]
        aliases = list(getattr(SPEC, "last_tier_aliases", []) or [])
        required_terms = list(getattr(SPEC, "required_terms", []) or [])
        keywords = [last_tier] + aliases + required_terms
        keywords = [k for k in keywords if k]
        if keywords:
            haystack = _haystack(item, qb_final)
            if not any(k.lower() in haystack for k in keywords):
                return False

    # 3. Optional block-terms.
    block_terms = getattr(SPEC, "block_terms", None) if SPEC is not None else None
    if block_terms:
        haystack = _haystack(item, qb_final)
        for term in block_terms:
            if term and term.lower() in haystack:
                return False

    return True