from __future__ import annotations

import math, random, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt_lib.paper_model import pdf_to_text

# Lazy embedding helper -- wraps the repo's llm_client
def _embed_batch(texts: List[str]) -> Optional[List[List[float]]]:
    if not texts:
        return []
    try:
        from prompt_lib.llm_client import get_embedding_client, embedding_model_name
        client = get_embedding_client()
        model  = embedding_model_name()
        resp   = client.embeddings.create(input=texts, model=model)
        return [d.embedding for d in resp.data]
    except Exception as e:
        print(f"[schema][embed] embedding call failed: {e}")
        return None

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da  = math.sqrt(sum(x * x for x in a))
    db  = math.sqrt(sum(x * x for x in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)

# Keyword-overlap fallback
_TOK_RE = re.compile(r"[a-zA-Z0-9]+")

def _tokens(s: str, min_len: int = 4) -> List[str]:
    return [t for t in _TOK_RE.findall((s or "").lower()) if len(t) >= min_len]

def _keyword_overlap_score(paper_text: str, target_keywords: List[str]) -> float:
    """Counts how many target tokens appear in the paper, normalized."""
    if not target_keywords:
        return 0.0
    body = (paper_text or "").lower()
    hits = sum(1 for kw in target_keywords if kw.lower() in body)
    return hits / max(1, len(target_keywords))

# Pure PDF loading
def load_pdf_texts(
    pdf_dir: Path,
    relevance_chars: int = 4000,
    proposal_chars: int  = 8000,
) -> List[Dict[str, str]]:
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not all_pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    out: List[Dict[str, str]] = []
    for pdf in all_pdfs:
        try:
            txt_path = pdf_to_text(pdf)
            text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
            out.append({
                "source_file":         pdf.name,
                "text_for_relevance":  text[:relevance_chars],
                "text_for_proposal":   text[:proposal_chars],
            })
        except Exception as e:
            print(f"  [warn] failed to read {pdf.name}: {e}")
    return out

# Score + filter + sample
def relevance_filtered_sample(
    papers: List[Dict[str, str]],
    last_tier_text: str,
    extra_keywords: Optional[List[str]] = None,
    max_papers: int = 20,
    drop_fraction: float = 0.25,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    extra_keywords = extra_keywords or []

    # Build a "query" text: last_tier_text + extra keywords.
    query_text = " ".join([last_tier_text, *extra_keywords]).strip()

    # Try embeddings first.
    method = "embedding"
    paper_texts = [p["text_for_relevance"] for p in papers]
    qvecs       = _embed_batch([query_text])
    if qvecs is None:
        scores: List[float] = []
        method = "keyword_fallback"
    else:
        pvecs = _embed_batch(paper_texts)
        if pvecs is None or len(pvecs) != len(papers):
            scores = []
            method = "keyword_fallback"
        else:
            scores = [_cosine(qvecs[0], pv) for pv in pvecs]

    if method == "keyword_fallback":
        # Build a small target-token bag from last_tier_text + extra_keywords.
        target_kws: List[str] = []
        for s in [last_tier_text, *extra_keywords]:
            for tok in _tokens(s, min_len=3):
                if tok not in target_kws:
                    target_kws.append(tok)
        print(f"[schema][relevance] embeddings unavailable -> "
              f"falling back to keyword overlap on {len(target_kws)} tokens")
        scores = [_keyword_overlap_score(p["text_for_relevance"], target_kws)
                  for p in papers]

    # Rank papers by score desc.
    ranked = sorted(zip(papers, scores), key=lambda kv: kv[1], reverse=True)

    n = len(ranked)
    # How many to drop from the bottom.
    drop_n = int(round(drop_fraction * n))
    keep_pool = ranked[: n - drop_n] if drop_n > 0 else list(ranked)

    # From the top-75% pool, sample up to max_papers (preserve relevance order
    # for determinism, but shuffle within to add variety).
    if len(keep_pool) > max_papers:
        # Take the top half deterministically, then random-fill the rest
        top_half = keep_pool[: max_papers // 2]
        rest_pool = keep_pool[max_papers // 2 :]
        rng.shuffle(rest_pool)
        chosen_pool = top_half + rest_pool[: max_papers - len(top_half)]
    else:
        chosen_pool = keep_pool

    chosen_files = {p["source_file"] for p, _ in chosen_pool}

    debug_rows: List[Dict[str, Any]] = []
    for rank, (p, sc) in enumerate(ranked, start=1):
        kept = p["source_file"] in chosen_files
        dropped_low = (rank > n - drop_n) if drop_n > 0 else False
        debug_rows.append({
            "source_file":          p["source_file"],
            "relevance_score":      round(float(sc), 6),
            "rank":                 rank,
            "kept":                 kept,
            "dropped_low_relevance": dropped_low,
            "method":               method,
        })

    selected = [p for p, _ in chosen_pool]
    print(f"[schema][relevance] method={method} "
          f"input={n} drop_low_25%={drop_n} pool={len(keep_pool)} "
          f"selected={len(selected)}")
    return selected, debug_rows
