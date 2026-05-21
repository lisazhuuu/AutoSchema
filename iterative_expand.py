from __future__ import annotations

import argparse, json, math, random, re, shutil, subprocess, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Domain constraints 
DOMAIN_LAST_TIERS: Dict[str, Dict[str, Any]] = {
    "chem": {"valid": ["cof", "mof", "zif"], "default": None},
    "ad":   {"valid": ["amyloid"],            "default": "amyloid"},
    "cs":   {"valid": ["prompt"],             "default": "prompt"},
}

LAST_TIER_TEXT: Dict[str, str] = {
    "cof":     "Covalent Organic Frameworks (COFs)",
    "mof":     "Metal-Organic Frameworks (MOFs)",
    "zif":     "Zeolitic Imidazolate Frameworks (ZIFs)",
    "amyloid": "Amyloid production and APP processing",
    "prompt":  "Prompt engineering for large language models",
}

SCHEMA_INDUCTION_DOMAINS: Set[str] = {"chem", "ad"}

def _resolve_domain(args) -> Tuple[str, str]:
    """Returns (last_tier_key, last_tier_text). Same contract as the frozen modules."""
    if args.domain not in DOMAIN_LAST_TIERS:
        raise SystemExit(
            f"--domain '{args.domain}' is not supported. "
            f"Valid: {sorted(DOMAIN_LAST_TIERS.keys())}"
        )
    entry = DOMAIN_LAST_TIERS[args.domain]
    chosen = (args.last_tier or entry["default"] or "").lower() or None
    if not chosen:
        raise SystemExit(
            f"--last_tier is required for --domain {args.domain}. "
            f"Valid options: {entry['valid']}"
        )
    if chosen not in entry["valid"]:
        raise SystemExit(
            f"--last_tier '{chosen}' is not valid for --domain {args.domain}. "
            f"Valid options: {entry['valid']}"
        )
    return chosen, LAST_TIER_TEXT[chosen]

# Small helpers
def _record_key(rec: Dict[str, Any]) -> Optional[str]:
    """Same priority as the paper: DOI -> source ID -> normalized title."""
    if not isinstance(rec, dict):
        return None
    doi = rec.get("doi")
    if doi:
        return str(doi).lower().strip()
    rid = rec.get("id")
    if rid:
        return str(rid).lower().strip()
    title = rec.get("title") or ""
    if title:
        return re.sub(r"\s+", " ", title.strip().lower())
    return None

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _run_subproc(args: List[str], *, env=None) -> None:
    """Run a python subprocess using the orchestrator's own interpreter."""
    cmd = [sys.executable] + args
    print(f"[orch] $ {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise SystemExit(
            f"[orch] subprocess failed (exit {res.returncode}): {' '.join(cmd)}"
        )
        
# Seed sampling with deterministic cap
def _cap_seeds(seed_dir: Path, max_seed_pdfs: int, seed_rng: int) -> List[Path]:
    pdfs = sorted(seed_dir.glob("*.pdf"))
    if not pdfs:
        return []
    if len(pdfs) <= max_seed_pdfs:
        return pdfs
    rng = random.Random(seed_rng)
    return sorted(rng.sample(pdfs, max_seed_pdfs))

# Next-round seed refresh: pick top-N by last-tier keyword overlap
_TOK_RE = re.compile(r"[a-zA-Z0-9]+")

def _last_tier_tokens(last_tier_text: str) -> List[str]:
    return [t.lower() for t in _TOK_RE.findall(last_tier_text or "") if len(t) >= 4]

def _rec_text_for_scoring(rec: Dict[str, Any]) -> str:
    return " ".join([
        rec.get("title") or "",
        rec.get("abstract") or "",
        rec.get("description") or "",
        (rec.get("expected_filename") or ""),
    ]).strip()

def _keyword_boost(text: str, tier_tokens: List[str]) -> float:
    if not tier_tokens or not text:
        return 0.0
    body = text.lower()
    hits = sum(1 for t in tier_tokens if t in body)
    return min(0.2, 0.05 * hits)

# Cosine on plain Python lists
def _cosine(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da  = math.sqrt(sum(x * x for x in a))
    db  = math.sqrt(sum(x * x for x in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)

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
        print(f"[orch][seed-refresh] embedding call failed: {e}")
        return None

def _score_next_seeds(
    accepted: List[Dict[str, Any]],
    last_tier_text: str,
    extra_query_text: str = "",
) -> Tuple[List[Dict[str, Any]], str]:
    tier_tokens = _last_tier_tokens(last_tier_text)
    method = "embedding"

    # Filter to records whose PDF actually exists on disk.
    usable: List[Tuple[Dict[str, Any], Path, str]] = []
    for rec in accepted:
        path = Path(rec.get("local_pdf_path") or "")
        if not path.exists():
            continue
        text = _rec_text_for_scoring(rec)
        if not text:
            text = path.name
        usable.append((rec, path, text))

    if not usable:
        return [], method

    # ---- Try embeddings first ----
    query_text = (last_tier_text + " " + extra_query_text).strip()
    qvecs = _embed_batch([query_text]) if query_text else None
    if qvecs is None:
        method = "keyword_fallback"
        cand_vecs = None
    else:
        cand_vecs = _embed_batch([t for _r, _p, t in usable])
        if cand_vecs is None or len(cand_vecs) != len(usable):
            method = "keyword_fallback"
            cand_vecs = None

    if method == "keyword_fallback":
        print(f"[orch][seed-refresh] embeddings unavailable -> falling back to "
              f"last-tier keyword overlap on {len(tier_tokens)} tokens")

    rows: List[Dict[str, Any]] = []
    for i, (rec, path, text) in enumerate(usable):
        kw_boost = _keyword_boost(text, tier_tokens)
        if cand_vecs is None:
            # Keyword-only fallback: re-use boost as the whole score so the
            # final_score column is still meaningful.
            emb_sim = 0.0
            final_score = kw_boost
        else:
            emb_sim = _cosine(qvecs[0], cand_vecs[i])
            final_score = emb_sim + kw_boost
        rows.append({
            "source_file":       path.name,
            "title":             rec.get("title"),
            "embedding_score":   round(float(emb_sim), 6),
            "keyword_boost":     round(float(kw_boost), 6),
            "final_score":       round(float(final_score), 6),
            "_path":             str(path),
        })
    return rows, method

def _pick_next_seeds(
    accepted: List[Dict[str, Any]],
    cumulative_corpus_dir: Path,
    last_tier_text: str,
    n_next: int,
    extra_query_text: str = "",
    debug_path: Optional[Path] = None,
) -> List[Path]:
    scored, method = _score_next_seeds(accepted, last_tier_text, extra_query_text)
    scored_sorted = sorted(scored, key=lambda r: r["final_score"], reverse=True)

    out: List[Path] = [Path(r["_path"]) for r in scored_sorted[:n_next]]
    chosen_names = {p.name for p in out}

    # Cumulative-corpus fallback (deterministic alphabetical order).
    filler_records: List[Dict[str, Any]] = []
    if len(out) < n_next:
        for p in sorted(cumulative_corpus_dir.glob("*.pdf")):
            if p.name in chosen_names:
                continue
            out.append(p)
            chosen_names.add(p.name)
            filler_records.append({
                "source_file":     p.name,
                "title":           None,
                "embedding_score": 0.0,
                "keyword_boost":   0.0,
                "final_score":     0.0,
                "_path":           str(p),
                "source":          "cumulative_corpus_fallback",
            })
            if len(out) >= n_next:
                break

    # Debug dump
    if debug_path is not None:
        debug_payload = {
            "method": method,
            "last_tier_text": last_tier_text,
            "extra_query_text": extra_query_text,
            "n_next": n_next,
            "selected_files": [p.name for p in out],
            "candidates": [{k: v for k, v in r.items() if k != "_path"}
                           for r in scored_sorted],
            "fillers": [{k: v for k, v in r.items() if k != "_path"}
                        for r in filler_records],
        }
        debug_path.write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return out

# Step 1: query generation (prompt_pipeline.py)
def _do_query_gen(
    *, domain: str, last_tier_key: str,
    seeds_input: Path, round_dir: Path, max_queries: int,
) -> Path:
    work = round_dir / "_qgen_work"
    work.mkdir(parents=True, exist_ok=True)
    _run_subproc([
        "prompt_pipeline.py",
        "--domain", domain,
        "--last_tier", last_tier_key,
        "--seeds", str(seeds_input),
        "--output", str(work),
        "--max_queries", str(max_queries),
    ])
    queries_path = work / "queries.json"
    if not queries_path.exists():
        raise SystemExit(f"[orch] queries.json not produced at {queries_path}")
    # Promote to round_dir for visibility.
    dest = round_dir / "queries.json"
    shutil.copy(queries_path, dest)
    return dest

# Step 2: retrieval (expand_round.py)
def _do_retrieval(
    *, domain: str, last_tier_key: str, queries_path: Path,
    round_dir: Path, results_per_source: int,
    sources_override: Optional[str],
) -> Tuple[Path, Path, Path]:
    """Returns (search_results.jsonl, downloaded_metadata.jsonl, downloads/) inside a work dir."""
    work = round_dir / "_retrieve_work"
    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        "expand_round.py",
        "--domain", domain,
        "--last_tier", last_tier_key,
        "--queries", str(queries_path),
        "--output", str(work),
        "--results_per_source", str(results_per_source),
    ]
    if sources_override:
        cmd += ["--sources", sources_override]
    _run_subproc(cmd)
    sr = work / "search_results.jsonl"
    dm = work / "downloaded_metadata.jsonl"
    dl = work / "downloads"
    return sr, dm, dl

# Step 3: dedup + accept
# ---------------------------------------------------------------------------
# Splits an attempt's downloaded_metadata.jsonl into THREE buckets:
#   new_unique:  not in history (cross-round-new) and not in round_accepted
#   overlap:     already in history, but NOT in round_accepted (we may use a
#                small number of these as fallback to reach min_accept)
#   dup_count:   rows skipped (within-round dup OR no key OR no PDF on disk)
# Within-round dedup is strict -- the same key is never returned twice.
# ---------------------------------------------------------------------------
def _dedup_accept(
    downloaded_metadata_path: Path,
    downloads_dir: Path,
    history_keys: Set[str],
    round_accepted_keys: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    if round_accepted_keys is None:
        round_accepted_keys = set()

    new_unique: List[Dict[str, Any]] = []
    overlap:    List[Dict[str, Any]] = []
    dup_count = 0
    local_seen: Set[str] = set()

    for rec in _read_jsonl(downloaded_metadata_path):
        key = _record_key(rec)
        if not key:
            continue
        # Strict within-round dedup: never accept the same key twice in one
        # round, and skip anything already accepted earlier in the round.
        if key in round_accepted_keys or key in local_seen:
            dup_count += 1
            continue

        # Resolve local PDF path.
        path = Path(rec.get("local_pdf_path") or "")
        if not path.exists():
            fname = rec.get("expected_filename") or Path(rec.get("local_pdf_path") or "").name
            if fname:
                alt = downloads_dir / fname
                if alt.exists():
                    rec["local_pdf_path"] = str(alt)
                    path = alt
        if not path.exists():
            # Metadata listed it but the file isn't on disk -- skip.
            continue
        local_seen.add(key)
        if key in history_keys:
            overlap.append(rec)        # already seen in a prior round
        else:
            new_unique.append(rec)     # cross-round-new
    return new_unique, overlap, dup_count

# Step 4: top-off (single attempt)
# ---------------------------------------------------------------------------
# Threshold-based top-off: trigger only when accepted < ceil(target_per_round
# * min_round_accept_ratio). Each attempt is budgeted from overfetch_need
# (not blindly results_per_source*2), so we don't over-retrieve.
# Multiple attempts are driven by the outer loop in `run_one_round`.
# ---------------------------------------------------------------------------
_DEFAULT_SOURCES_BY_DOMAIN: Dict[str, Tuple[str, ...]] = {
    "chem": ("arxiv", "openalex", "nature", "core"),
    "ad":   ("pubmed", "biorxiv", "openalex", "core"),
    "cs":   ("arxiv", "openalex"),
}


def _active_source_count(domain: str, sources_override: Optional[str]) -> int:
    if sources_override:
        n = sum(1 for s in sources_override.split(",") if s.strip())
        return max(1, n)
    return max(1, len(_DEFAULT_SOURCES_BY_DOMAIN.get(domain, ("arxiv",))))


def _topoff_budget_per_source(topoff_need: int, source_count: int,
                              base_depth: int,
                              buffer: int = 3, minimum: int = 2) -> int:
    """Top-off per-source budget that goes DEEPER than the main pass:
        base_depth + ceil(need / source_count) + buffer
    so we don't just refetch the same top-K from each API.

    `base_depth` is the per-source depth the main retrieval used in this
    round (after any --round_depth_increment ramp). `buffer` defaults to 3
    to absorb dedup/download churn. `minimum` is a hard floor.
    """
    if source_count <= 0:
        return max(minimum, base_depth)
    if topoff_need <= 0:
        return max(minimum, base_depth)
    return max(minimum, base_depth + math.ceil(topoff_need / source_count) + buffer)


def _do_topoff_attempt(
    *, domain: str, last_tier_key: str, queries_path: Path,
    work_dir: Path, sources_override: Optional[str],
    topoff_need: int, source_count: int,
    base_depth: int,
    history_keys: Set[str], round_accepted_keys: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One top-off attempt. Returns (new_unique, overlap_candidates):
        new_unique   -- not in history_keys / round_accepted_keys
        overlap_pool -- already in history_keys, NOT in round_accepted_keys
    Both are capped at `topoff_need` items each (we don't return more than
    the round could plausibly want)."""
    if topoff_need <= 0:
        return [], []

    per_source = _topoff_budget_per_source(topoff_need, source_count, base_depth)
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "expand_round.py",
        "--domain", domain,
        "--last_tier", last_tier_key,
        "--queries", str(queries_path),
        "--output", str(work_dir),
        "--results_per_source", str(per_source),
    ]
    if sources_override:
        cmd += ["--sources", sources_override]
    print(f"[orch][topoff] attempt budget: need={topoff_need}  "
          f"base_depth={base_depth}  per_source={per_source} "
          f"(across {source_count} sources)")
    _run_subproc(cmd)

    new_unique, overlap, _dup = _dedup_accept(
        work_dir / "downloaded_metadata.jsonl",
        work_dir / "downloads",
        history_keys=history_keys,
        round_accepted_keys=round_accepted_keys,
    )
    return new_unique[:topoff_need], overlap[:topoff_need]

def _append_topoff_to_round_logs(
    *, round_dir: Path, topoff_work_dir: Path, extra_accepted: List[Dict[str, Any]],
) -> None:
    extra_keys = {_record_key(r) for r in extra_accepted if _record_key(r)}
    if not extra_keys:
        return

    # Append matching rows from top-off search_results.jsonl (if present).
    topoff_sr = topoff_work_dir / "search_results.jsonl"
    if topoff_sr.exists():
        kept_sr: List[Dict[str, Any]] = []
        for rec in _read_jsonl(topoff_sr):
            k = _record_key(rec)
            if k and k in extra_keys:
                kept_sr.append(rec)
        if kept_sr:
            with (round_dir / "search_results.jsonl").open("a", encoding="utf-8") as f:
                for rec in kept_sr:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Append the accepted top-off rows to downloaded_metadata.jsonl directly.
    with (round_dir / "downloaded_metadata.jsonl").open("a", encoding="utf-8") as f:
        for rec in extra_accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Step 5: schema induction (design_extraction_prompt.py)
def _do_schema(
    *, domain: str, last_tier_key: str, pdf_dir: Path, schema_dir: Path,
    prev_schema: Optional[Path],
    max_schema_papers: int, max_chars_per_paper: int,
    max_final_fields: int, min_support: int,
) -> None:
    schema_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "design_extraction_prompt.py",
        "--domain", domain,
        "--last_tier", last_tier_key,
        "--pdf_dir", str(pdf_dir),
        "--output", str(schema_dir),
        "--max_papers", str(max_schema_papers),
        "--max_chars_per_paper", str(max_chars_per_paper),
        "--max_final_fields", str(max_final_fields),
        "--min_support", str(min_support),
    ]
    if prev_schema:
        cmd += ["--prev_schema", str(prev_schema)]
    _run_subproc(cmd)

# Round driver
def run_one_round(
    *, round_idx: int, args, last_tier_key: str, last_tier_text: str,
    seed_dir: Path, prev_schema: Optional[Path],
    history_keys: Set[str], output_root: Path,
) -> Dict[str, Any]:
    round_dir = output_root / f"round_{round_idx:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== ROUND {round_idx} ==========")
    print(f"[orch] round_dir   = {round_dir}")
    print(f"[orch] seed_dir    = {seed_dir}")
    print(f"[orch] prev_schema = {prev_schema}")

    # ---- 1. Stage seeds (cap at max_seed_pdfs) ----
    seeds = _cap_seeds(seed_dir, args.max_seed_pdfs, args.seed)
    if not seeds:
        raise SystemExit(f"[orch] no seed PDFs in {seed_dir}")
    seeds_input = round_dir / "seeds_input"
    seeds_input.mkdir(exist_ok=True)
    for p in seeds:
        target = seeds_input / p.name
        if not target.exists():
            shutil.copy(p, target)
    print(f"[orch] staged {len(seeds)} seed PDFs -> {seeds_input}")

    # ---- 2. Query generation ----
    queries_path = _do_query_gen(
        domain=args.domain, last_tier_key=last_tier_key,
        seeds_input=seeds_input, round_dir=round_dir,
        max_queries=args.max_queries,
    )

    # ---- 3. Retrieval (depth ramps up each round) ----
    # round_results_per_source = results_per_source + (round_idx-1)*increment
    # so later rounds search DEEPER instead of refetching the same shallow
    # top-K (which would mostly be cross-round duplicates).
    round_results_per_source = (args.results_per_source
                                + (round_idx - 1) * args.round_depth_increment)
    print(f"[orch] round_results_per_source={round_results_per_source} "
          f"(base={args.results_per_source} + "
          f"(round {round_idx}-1)*{args.round_depth_increment})")
    _sr_path, dm_path, dl_dir = _do_retrieval(
        domain=args.domain, last_tier_key=last_tier_key,
        queries_path=queries_path, round_dir=round_dir,
        results_per_source=round_results_per_source,
        sources_override=args.sources,
    )
    # Promote artifacts to round_dir.
    for name in ("search_results.jsonl", "downloaded_metadata.jsonl"):
        src = round_dir / "_retrieve_work" / name
        if src.exists():
            shutil.copy(src, round_dir / name)

    # ---- 4. Dedup (split: new_unique vs cross-round overlap) ----
    target_per_round    = max(1, math.ceil(args.target_total_papers / max(1, args.rounds)))
    min_accept          = max(1, math.ceil(target_per_round * args.min_round_accept_ratio))
    overfetch_target    = max(target_per_round,
                              math.ceil(target_per_round * args.topoff_overfetch_ratio))
    max_allowed_overlap = math.floor(target_per_round * args.overlap_threshold)
    source_count        = _active_source_count(args.domain, args.sources)

    new_main, overlap_main, dup_count = _dedup_accept(
        dm_path, dl_dir, history_keys=history_keys, round_accepted_keys=set())

    # accepted_new = cross-round-new papers accepted this round (priority a + b).
    # overlap_pool = cross-round duplicates we MAY use as fallback (priority c).
    accepted_new: List[Dict[str, Any]] = list(new_main)
    overlap_pool: List[Dict[str, Any]] = list(overlap_main)

    print(f"[orch] dedup: new_unique={len(new_main)} overlap_candidates="
          f"{len(overlap_main)} within_round_or_missing_dups={dup_count}")
    print(f"[orch] target_per_round={target_per_round}  "
          f"min_accept={min_accept} (min_round_accept_ratio={args.min_round_accept_ratio})  "
          f"overfetch_target={overfetch_target} "
          f"(topoff_overfetch_ratio={args.topoff_overfetch_ratio})")
    print(f"[orch] overlap_threshold={args.overlap_threshold}  "
          f"max_allowed_overlap={max_allowed_overlap}  "
          f"history_keys(unique)={len(history_keys)}/{args.target_total_papers}")

    # ---- 5. Deeper top-off (only counts NEW unique toward the trigger) ----
    topoff_warning = False
    if len(accepted_new) < min_accept:
        # round_accepted_keys = history + this-round accepted-new keys, updated
        # after each attempt so a later attempt can't re-accept earlier picks.
        round_accepted_keys: Set[str] = {k for k in
                                         (_record_key(r) for r in accepted_new) if k}

        for attempt in range(1, args.max_topoff_attempts + 1):
            remaining_total = args.target_total_papers - (len(history_keys)
                                                          + len(accepted_new))
            overfetch_need = max(0, overfetch_target - len(accepted_new))
            attempt_need = min(overfetch_need, max(0, remaining_total) + 2)
            print(f"[orch][topoff] attempt {attempt}/{args.max_topoff_attempts}: "
                  f"accepted_new={len(accepted_new)}  overfetch_target={overfetch_target}  "
                  f"overfetch_need={overfetch_need}  remaining_total={remaining_total}  "
                  f"attempt_need={attempt_need}")
            if attempt_need <= 0:
                break

            attempt_work_dir = round_dir / f"_topoff_work_attempt_{attempt:02d}"
            extra_new, extra_overlap = _do_topoff_attempt(
                domain=args.domain, last_tier_key=last_tier_key,
                queries_path=queries_path,
                work_dir=attempt_work_dir,
                sources_override=args.sources,
                topoff_need=attempt_need,
                source_count=source_count,
                base_depth=round_results_per_source,
                history_keys=history_keys,
                round_accepted_keys=round_accepted_keys,
            )
            print(f"[orch][topoff] attempt {attempt}: added {len(extra_new)} "
                  f"new-unique papers ({len(extra_overlap)} overlap candidates seen)")

            accepted_new.extend(extra_new)
            for rec in extra_new:
                k = _record_key(rec)
                if k:
                    round_accepted_keys.add(k)
            # New overlap candidates that aren't already pooled.
            pooled_overlap_keys = {k for k in (_record_key(r) for r in overlap_pool) if k}
            for rec in extra_overlap:
                k = _record_key(rec)
                if k and k not in pooled_overlap_keys:
                    overlap_pool.append(rec)
                    pooled_overlap_keys.add(k)
            if extra_new:
                _append_topoff_to_round_logs(
                    round_dir=round_dir,
                    topoff_work_dir=attempt_work_dir,
                    extra_accepted=extra_new,
                )

            if len(accepted_new) >= overfetch_target:
                break
            if not extra_new:
                print(f"[orch][topoff] attempt {attempt} yielded 0 new-unique "
                      "papers; skipping further attempts.")
                break

        if len(accepted_new) >= min_accept:
            print(f"[orch][topoff] min_accept reached with new uniques: "
                  f"accepted_new={len(accepted_new)} >= min_accept={min_accept}")
    else:
        print(f"[orch][topoff] no top-off needed "
              f"(accepted_new={len(accepted_new)} >= min_accept={min_accept})")

    # ---- 5b. Build final accepted list: new uniques first, then overlap ----
    # Cap NEW uniques by the global remaining target so cumulative unique count
    # never exceeds target_total_papers. Overlap papers are already in history,
    # so they do NOT consume the global target -- they only fill the round.
    room_for_unique = max(0, args.target_total_papers - len(history_keys))
    unique_cap = min(target_per_round, room_for_unique)
    chosen_unique = accepted_new[:unique_cap]

    overlap_slots = max(0, min(target_per_round - len(chosen_unique),
                               max_allowed_overlap))
    chosen_overlap = overlap_pool[:overlap_slots]

    accepted = chosen_unique + chosen_overlap
    print(f"[orch] accepted breakdown: new_unique_used={len(chosen_unique)}  "
          f"overlap_used={len(chosen_overlap)}/{max_allowed_overlap}  "
          f"final_accepted={len(accepted)}  (unique_cap={unique_cap})")

    if len(accepted) < min_accept and len(accepted) > 0:
        topoff_warning = True
        print(f"[orch][topoff] WARNING: final accepted={len(accepted)} < "
              f"min_accept={min_accept} even after deeper top-off + "
              f"{len(chosen_overlap)} allowed overlap; continuing because "
              f"usable PDFs and next seeds may still exist.")

    # ---- 6. Materialize accepted PDFs into round_dir/downloads/ + cumulative_corpus ----
    round_downloads = round_dir / "downloads"
    round_downloads.mkdir(exist_ok=True)
    cumulative_corpus = output_root / "cumulative_corpus"
    cumulative_corpus.mkdir(exist_ok=True)

    final_accepted: List[Dict[str, Any]] = []
    for rec in accepted:
        src = Path(rec.get("local_pdf_path") or "")
        if not src.exists():
            continue
        round_pdf = round_downloads / src.name
        if not round_pdf.exists():
            shutil.copy(src, round_pdf)
        rec["local_pdf_path"] = str(round_pdf)

        cum_pdf = cumulative_corpus / src.name
        if not cum_pdf.exists():
            shutil.copy(src, cum_pdf)

        final_accepted.append(rec)

    # ---- 7. Update history_keys + cumulative_metadata.jsonl ----
    cumulative_metadata = output_root / "cumulative_metadata.jsonl"
    new_for_cum: List[Dict[str, Any]] = []
    for rec in final_accepted:
        key = _record_key(rec)
        if not key or key in history_keys:
            continue
        history_keys.add(key)
        new_for_cum.append(rec)
    if new_for_cum:
        _append_jsonl(cumulative_metadata, new_for_cum)

    # Per-round manifests of the FINAL accepted set (new uniques + allowed
    # overlap, after main retrieval + deeper top-off + dedup + trim). This is
    # the authoritative "what did this round contribute" without scanning
    # _retrieve_work / _topoff_work_attempt_*.
    with (round_dir / "accepted_metadata.jsonl").open("w", encoding="utf-8") as f:
        for r in final_accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (round_dir / "final_downloaded_metadata.jsonl").open("w", encoding="utf-8") as f:
        for r in final_accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 8. Schema induction ----
    schema_dir = round_dir / "schema"
    final_schema_path: Optional[Path] = None
    if args.domain in SCHEMA_INDUCTION_DOMAINS and final_accepted:
        _do_schema(
            domain=args.domain, last_tier_key=last_tier_key,
            pdf_dir=round_downloads, schema_dir=schema_dir,
            prev_schema=prev_schema,
            max_schema_papers=args.max_schema_papers,
            max_chars_per_paper=args.max_chars_per_paper,
            max_final_fields=args.max_final_fields,
            min_support=args.min_support,
        )
        final_schema_path = schema_dir / "final_schema.json"
        if not final_schema_path.exists():
            final_schema_path = None
    elif args.domain not in SCHEMA_INDUCTION_DOMAINS:
        print(f"[orch] --domain {args.domain} -> schema induction intentionally "
              f"SKIPPED in the KDD artifact.")
    else:
        print("[orch] no accepted PDFs this round -> schema induction skipped.")

    # ---- 9. Next-round seed refresh ----
    next_seeds_dir = round_dir / "next_seeds"
    next_seeds_dir.mkdir(exist_ok=True)

    extra_query_text = ""
    try:
        q_data = json.loads((round_dir / "queries.json").read_text(encoding="utf-8"))
        if isinstance(q_data, dict):
            q_data = q_data.get("queries", [])
        chunks: List[str] = []
        for q in (q_data or [])[:5]:
            if isinstance(q, dict):
                chunks.append(q.get("query_nl") or q.get("query_bool") or "")
        extra_query_text = " ".join(c for c in chunks if c).strip()
    except Exception:
        extra_query_text = ""

    picked = _pick_next_seeds(
        accepted=final_accepted,
        cumulative_corpus_dir=cumulative_corpus,
        last_tier_text=last_tier_text,
        n_next=args.seeds_per_next_round,
        extra_query_text=extra_query_text,
        debug_path=round_dir / "seed_refresh_scores.json",
    )
    for p in picked:
        target = next_seeds_dir / p.name
        if not target.exists():
            shutil.copy(p, target)
    print(f"[orch] next_seeds  = {len(picked)} PDFs -> {next_seeds_dir} "
          f"(scoring in {round_dir / 'seed_refresh_scores.json'})")

    return {
        "round_dir":               round_dir,
        "schema_dir":              schema_dir if final_schema_path else None,
        "final_schema_path":       final_schema_path,
        "next_seeds_dir":          next_seeds_dir,
        "accepted":                len(final_accepted),
        "duplicates":              dup_count,
        "topoff_warning":          topoff_warning,
    }

# CLI
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AutoSchema Phase-I iterative orchestrator (KDD camera-ready artifact)."
    )
    p.add_argument("--domain", required=True, choices=list(DOMAIN_LAST_TIERS.keys()),
                   help="Active domain. KDD artifact supports chem | ad | cs.")
    p.add_argument("--last_tier", "--last-tier",
                   dest="last_tier", default=None,
                   choices=["cof", "mof", "zif", "amyloid", "prompt"],
                   help="Required for chem (cof|mof|zif); defaults for ad/cs.")
    p.add_argument("--seed_dir", required=True,
                   help="Folder of seed PDFs used in round 1.")
    p.add_argument("--output_root", default=None,
                   help="Top-level run folder. Defaults to "
                        "runs/<domain>/<last_tier>/iterative_phase1.")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max_queries", type=int, default=8)
    p.add_argument("--results_per_source", type=int, default=7,
                   help="Base per-source retrieval depth for round 1. Later "
                        "rounds add (round_idx-1)*round_depth_increment.")
    p.add_argument("--round_depth_increment", type=int, default=5,
                   help="Per-round increase in retrieval depth: "
                        "round_results_per_source = results_per_source + "
                        "(round_idx-1)*round_depth_increment. Default 5.")
    p.add_argument("--target_total_papers", type=int, default=50,
                   help="Target unique accepted papers across all rounds. "
                        "Per-round target = ceil(total / rounds).")
    p.add_argument("--min_round_accept_ratio", type=float, default=0.80,
                   help="Trigger top-off when accepted_new < "
                        "ceil(target_per_round * min_round_accept_ratio). "
                        "Default 0.80.")
    p.add_argument("--topoff_overfetch_ratio", type=float, default=1.20,
                   help="Top-off overfetches toward "
                        "ceil(target_per_round * topoff_overfetch_ratio) "
                        "before the final trim. Default 1.20.")
    p.add_argument("--overlap_threshold", type=float, default=0.15,
                   help="Allow up to floor(target_per_round * overlap_threshold) "
                        "cross-round overlap papers per round as a FALLBACK "
                        "(only used if new uniques can't reach min_accept). "
                        "cumulative_metadata.jsonl stays unique-only. "
                        "Default 0.15.")
    p.add_argument("--max_topoff_attempts", type=int, default=2,
                   help="Maximum top-off retries per round. Default 2. If still "
                        "below min_accept after this many attempts (even with "
                        "allowed overlap), a warning is printed and the run "
                        "continues -- it is NOT a hard early stop.")
    p.add_argument("--seeds_per_next_round", type=int, default=5)
    p.add_argument("--max_seed_pdfs", type=int, default=10,
                   help="Cap on seed PDFs used per round.")
    p.add_argument("--max_schema_papers", type=int, default=20)
    p.add_argument("--max_chars_per_paper", type=int, default=12000)
    p.add_argument("--max_final_fields", type=int, default=12)
    p.add_argument("--min_support", type=int, default=2)
    p.add_argument("--sources", default=None,
                   help="(Optional) CSV override for retrieval sources. "
                        "Defaults come from expand_lib.domain_args per domain.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for deterministic seed sampling.")
    return p

def main():
    args = _make_parser().parse_args()
    last_tier_key, last_tier_text = _resolve_domain(args)

    seed_dir = Path(args.seed_dir)
    if not seed_dir.exists():
        raise SystemExit(f"--seed_dir not found: {seed_dir}")

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = Path(f"runs/{args.domain}/{last_tier_key}/iterative_phase1")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cumulative_corpus").mkdir(exist_ok=True)
    cum_meta = output_root / "cumulative_metadata.jsonl"
    if not cum_meta.exists():
        cum_meta.touch()

    print(f"[orch] domain={args.domain}  last_tier={last_tier_key}  text={last_tier_text!r}")
    print(f"[orch] seed_dir={seed_dir}  output_root={output_root}")
    print(f"[orch] rounds={args.rounds}  target_total_papers={args.target_total_papers}")

    history_keys: Set[str] = set()
    # Restore history_keys if cumulative_metadata.jsonl already has entries
    # (supports resumed runs without re-counting prior accepts as new).
    for rec in _read_jsonl(cum_meta):
        k = _record_key(rec)
        if k:
            history_keys.add(k)
    if history_keys:
        print(f"[orch] resumed history_keys: {len(history_keys)} keys")

    prev_schema: Optional[Path] = None
    current_seed_dir = seed_dir
    last_successful_schema_dir: Optional[Path] = None

    for r in range(1, args.rounds + 1):
        info = run_one_round(
            round_idx=r, args=args,
            last_tier_key=last_tier_key, last_tier_text=last_tier_text,
            seed_dir=current_seed_dir, prev_schema=prev_schema,
            history_keys=history_keys, output_root=output_root,
        )

        if info.get("schema_dir") and info.get("final_schema_path"):
            last_successful_schema_dir = info["schema_dir"]
            prev_schema = info["final_schema_path"]

        # Top-off / overlap could not reach min_accept. This is a WARNING only,
        # NOT a hard stop -- per the KDD paper, only the three conditions below
        # stop the run.
        if info.get("topoff_warning"):
            print(f"[orch] round {r} produced a top-off warning "
                  f"(min_round_accept_ratio={args.min_round_accept_ratio} not "
                  f"reached even with up to "
                  f"floor(target*{args.overlap_threshold}) overlap); continuing "
                  f"because usable PDFs and next seeds may still exist.")

        # ---- True early-stop conditions ----
        # (a) zero usable PDFs this round even after top-off + overlap -> stop.
        if info.get("accepted", 0) <= 0:
            print(f"[orch] EARLY STOP: round {r} accepted 0 PDFs after dedup + top-off.")
            break

        # (b) hit cumulative target -> stop.
        if len(history_keys) >= args.target_total_papers:
            print(f"[orch] EARLY STOP: cumulative history {len(history_keys)} >= "
                  f"target_total_papers {args.target_total_papers}.")
            break

        # (c) carry next_seeds into the next round, or stop if empty.
        nseeds = info.get("next_seeds_dir")
        if nseeds and any(nseeds.glob("*.pdf")):
            current_seed_dir = nseeds
        else:
            print(f"[orch] EARLY STOP: round {r} produced no next-round seeds.")
            break

    # ---- Final freeze ----
    final_root = output_root / "final_schemas"
    final_root.mkdir(exist_ok=True)
    if last_successful_schema_dir is not None:
        sd = last_successful_schema_dir
        for name in ("final_schema.json", "extraction_prompt.txt"):
            src = sd / name
            if src.exists():
                shutil.copy(src, final_root / name)
        print(f"\n[orch] frozen schema/prompt -> {final_root}  "
              f"(from {sd})")
    else:
        notice = final_root / "README.txt"
        notice.write_text(
            "Schema induction was skipped for this run.\n"
            f"  domain={args.domain}\n"
            f"  reason: --domain '{args.domain}' is not in "
            f"SCHEMA_INDUCTION_DOMAINS ({sorted(SCHEMA_INDUCTION_DOMAINS)}) "
            "for the KDD artifact, OR no PDFs were accepted in the last round.\n",
            encoding="utf-8",
        )
        print(f"\n[orch] no frozen schema (see {notice})")

    print(f"\n[orch] cumulative corpus  : {output_root / 'cumulative_corpus'}")
    print(f"[orch] cumulative metadata: {cum_meta}")
    print(f"[orch] history_keys total : {len(history_keys)}")


if __name__ == "__main__":
    main()
