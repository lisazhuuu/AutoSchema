import argparse, json, os, re, time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from expand_lib.domain_args import (
    DOMAINS,
    LAST_TIERS,
    SEARCH_SOURCES,
    DomainSpec,
    get_spec,
    parse_sources,
)
from expand_lib.preprocessing import load_queries
from expand_lib.searches import (
    download_biorxiv_hit,
    download_pubmed_or_pmc_hit,
    filter_query_for_biorxiv,
    search_arxiv,
    search_biorxiv,
    search_core,
    search_crossref_nature,
    search_openalex,
    search_pubmed,
)
from expand_lib.utils import (
    dedupe,
    delete_empty_folders,
    delete_invalid_pdfs,
    download_pdf,
    fname_with_tag,
)

# ==================== Helpers ====================
def get_record_key(rec: dict) -> str | None:
    """DOI -> source ID -> normalized title."""
    if not rec:
        return None
    doi = rec.get("doi")
    if doi:
        return doi.lower().strip()
    rid = rec.get("id")
    if rid:
        return str(rid).lower().strip()
    title = rec.get("title") or ""
    if title:
        return re.sub(r"\s+", " ", title.strip().lower())
    return None

def _resolve_last_tier(args, spec: DomainSpec, queries: List[Dict]) -> DomainSpec:
    chosen = args.last_tier
    if not chosen and queries:
        q_tiers = (queries[0].get("tiers") or [])
        if q_tiers:
            text = (q_tiers[-1] or "").strip().lower()
            for key, lt in LAST_TIERS.items():
                if text == lt.text.lower() and key in spec.valid_last_tiers:
                    chosen = key
                    break
    return spec.resolve(chosen)

def _build_query(src: str, q: dict, domain_keyword: str) -> str:
    """Pick the right form of the query for each source."""
    qb = q.get("query_bool") or q.get("query_nl") or ""
    qn = q.get("query_nl")   or qb

    if domain_keyword:
        simplified = f'"{domain_keyword}" AND {qn}'.replace('"', '').replace("'", "")
    else:
        simplified = qn

    if src == "biorxiv":
        return filter_query_for_biorxiv(qn or simplified)
    if src in ("openalex", "nature", "core", "pubmed"):
        return simplified
    return qb

def _run_search(src: str, query: str, per_source: int, *,
                core_api_key: str | None,
                openalex_mailto: str | None,
                crossref_mailto: str | None,
                arxiv_cats: list,
                domain_spec) -> list:
    if src == "arxiv":
        return search_arxiv(query, per_source, arxiv_cats,
                            domain_spec=domain_spec) or []
    if src == "openalex":
        return search_openalex(query, per_source, mailto=openalex_mailto,
                               domain_spec=domain_spec) or []
    if src == "nature":
        return search_crossref_nature(query, per_source, mailto=crossref_mailto,
                                      domain_spec=domain_spec) or []
    if src == "biorxiv":
        return search_biorxiv(query, per_source, domain_spec=domain_spec) or []
    if src == "pubmed":
        return search_pubmed(query, per_source, mailto=openalex_mailto,
                             domain_spec=domain_spec) or []
    if src == "core":
        # CORE returns None on rate-limit; bounded retry.
        for attempt in range(3):
            hits = search_core(query, per_source,
                               api_key=core_api_key, domain_spec=domain_spec)
            if hits is None:
                wait = 5 * (attempt + 1)
                print(f"[CORE] Rate limited. Retry in {wait}s...")
                time.sleep(wait)
                continue
            time.sleep(1.0 if hits else 0.8)
            return hits or []
        return []
    print(f"[expand_round] Unknown source: {src}")
    return []

def _dl_score(r: Dict) -> tuple:
    host = (urlparse(r.get("pdf_url") or r.get("url") or "").hostname or "").lower()
    return (
        r.get("domain_score", 0),
        1 if r.get("pdf_url") else 0,
        1 if "arxiv.org" in host else 0,
    )

# ==================== Main ====================
def run_one_round_search_and_download(
    queries_path: Path,
    round_dir: Path,
    domain_spec: DomainSpec,
    sources: tuple,
    *,
    core_api_key: str | None = None,
    results_per_source: int = 7,
    contact_emails: Dict[str, str] = None,
    exclude_keys: Set[str] = None,
):
    if exclude_keys is None:
        exclude_keys = set()

    target_per_round = results_per_source * len(sources)
    candidate_budget = max(target_per_round * 3, 50)

    round_dir.mkdir(parents=True, exist_ok=True)
    download_dir = round_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(str(queries_path))

    crossref_mailto = ((contact_emails or {}).get("CROSSREF_MAILTO")
                       or os.getenv("CROSSREF_MAILTO"))
    openalex_mailto = ((contact_emails or {}).get("OPENALEX_MAILTO")
                       or os.getenv("OPENALEX_MAILTO")
                       or crossref_mailto)

    domain_keyword = domain_spec.tiers[-1] if domain_spec.tiers else ""

    # Candidate collection 
    candidate_map: Dict[str, Dict] = {}
    for q in queries:
        if len(candidate_map) >= candidate_budget:
            break
        for src in sources:
            if len(candidate_map) >= candidate_budget:
                break
            need = candidate_budget - len(candidate_map)
            per_source = min(results_per_source, need)
            query = _build_query(src, q, domain_keyword)

            try:
                hits = _run_search(
                    src, query, per_source,
                    core_api_key=core_api_key,
                    openalex_mailto=openalex_mailto,
                    crossref_mailto=crossref_mailto,
                    arxiv_cats=domain_spec.arxiv_cats,
                    domain_spec=domain_spec,
                )
            except Exception as e:
                print(f"[ERROR] {src} search failed for '{query[:50]}...': {e}")
                continue

            for h in hits:
                key = get_record_key(h)
                if not key or key in exclude_keys or key in candidate_map:
                    continue
                candidate_map[key] = h

    # Rank + select 
    all_hits = sorted(dedupe(list(candidate_map.values())),
                      key=_dl_score, reverse=True)
    for rec in all_hits:
        rec["expected_filename"] = fname_with_tag(rec)

    selected_for_metadata = all_hits[:target_per_round]
    search_results_path = round_dir / "search_results.jsonl"
    with open(search_results_path, "w", encoding="utf-8") as f:
        for r in selected_for_metadata:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Download 
    downloaded_paths: List[Path] = []
    downloaded_per_source: Dict[str, int] = {}
    attempted_keys: Set[str] = set()

    def _record_success(src_lower: str, path: Path) -> None:
        downloaded_paths.append(path)
        downloaded_per_source[src_lower] = downloaded_per_source.get(src_lower, 0) + 1

    def _try_download(rec: Dict) -> None:
        source = (rec.get("source") or "").lower()
        fname = fname_with_tag(rec)
        dest = download_dir / fname

        if dest.exists() and dest.stat().st_size > 1000:
            if dest not in downloaded_paths:
                _record_success(source, dest)
            return

        if source == "pubmed":
            if download_pubmed_or_pmc_hit(rec, download_dir, fname):
                if dest.exists() and dest.stat().st_size > 1000:
                    _record_success(source, dest)
            return

        if source == "biorxiv":
            if download_biorxiv_hit(rec, download_dir, fname):
                if dest.exists() and dest.stat().st_size > 1000:
                    _record_success(source, dest)
            return

        if rec.get("local_path") and os.path.exists(rec["local_path"]):
            _record_success(source, Path(rec["local_path"]))
            return

        pdf_guess = rec.get("pdf_url")
        if pdf_guess and download_pdf(pdf_guess, dest):
            if dest.exists() and dest.stat().st_size > 1000:
                _record_success(source, dest)
                return

    def _record_key(rec: Dict) -> str:
        return get_record_key(rec) or f"_obj_{id(rec)}"

    # Pass 1: per-source cap
    for rec in all_hits:
        if len(downloaded_paths) >= target_per_round:
            break
        source = (rec.get("source") or "").lower()
        if downloaded_per_source.get(source, 0) >= results_per_source:
            continue
        key = _record_key(rec)
        if key in attempted_keys:
            continue
        attempted_keys.add(key)
        _try_download(rec)

    pass1_per_source = dict(downloaded_per_source)
    print(f"[expand_round] Pass 1 (capped) per-source: {pass1_per_source} "
          f"-> {len(downloaded_paths)}/{target_per_round}")

    # Pass 2: overflow (no cap)
    if len(downloaded_paths) < target_per_round:
        for rec in all_hits:
            if len(downloaded_paths) >= target_per_round:
                break
            key = _record_key(rec)
            if key in attempted_keys:
                continue
            attempted_keys.add(key)
            _try_download(rec)
        overflow = {s: downloaded_per_source[s] - pass1_per_source.get(s, 0)
                    for s in downloaded_per_source
                    if downloaded_per_source[s] - pass1_per_source.get(s, 0) > 0}
        if overflow:
            print(f"[expand_round] Pass 2 (overflow) added: {overflow}")

    print(f"[expand_round] Final per-source downloads: {dict(downloaded_per_source)}")

    # Cleanup + final manifest 
    delete_invalid_pdfs(download_dir)
    delete_empty_folders(download_dir)

    final_pdf_paths = [p for p in downloaded_paths
                       if p.exists() and p.stat().st_size > 1000]
    final_names = {p.name for p in final_pdf_paths}

    downloaded_metadata: List[Dict] = []
    seen_filenames: Set[str] = set()
    for rec in all_hits:
        fname = rec.get("expected_filename")
        if fname and fname in final_names and fname not in seen_filenames:
            seen_filenames.add(fname)
            rec_out = dict(rec)
            rec_out["local_pdf_path"] = str(download_dir / fname)
            downloaded_metadata.append(rec_out)

    downloaded_metadata_path = round_dir / "downloaded_metadata.jsonl"
    with open(downloaded_metadata_path, "w", encoding="utf-8") as f:
        for r in downloaded_metadata:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "download_dir": download_dir,
        "selected_pdf_paths": final_pdf_paths,
        "search_results_path": search_results_path,
        "downloaded_metadata_path": downloaded_metadata_path,
    }

# ==================== CLI ====================
def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one expansion round: search + download PDFs from queries.json."
    )
    parser.add_argument("--queries", required=True,
                        help="Path to queries.json for this round")
    parser.add_argument("--output", default=None,
                        help="Round directory (default: runs/<domain>/test_expand_<ts>)")
    parser.add_argument(
        "--domain", required=True,
        choices=list(DOMAINS.keys()),         # ["chem", "cs", "ad"]
        help="Active domain. One of: chem | cs | ad.",
    )
    parser.add_argument(
        "--last_tier", "--last-tier",
        dest="last_tier",
        choices=list(LAST_TIERS.keys()),      # ["cof","mof","zif","prompt","amyloid"]
        default=None,
        help="Retrieval-focus last tier. "
             "REQUIRED for --domain chem (cof | mof | zif). "
             "Defaults to 'prompt' for cs, 'amyloid' for ad.",
    )
    parser.add_argument(
        "--sources", default=None,
        help="Comma-separated subset of: "
             f"{','.join(sorted(SEARCH_SOURCES.keys()))}. "
             "If omitted, uses the per-domain default from "
             "expand_lib.domain_args.DOMAINS.",
    )
    parser.add_argument("--results_per_source", type=int, default=7,
                        help="Per-source download cap (Pass 1).")
    parser.add_argument("--crossref_email", default=None,
                        help="Polite-pool email for Crossref. Falls back to "
                             "$CROSSREF_MAILTO.")
    parser.add_argument("--openalex_email", default=None,
                        help="Polite-pool email for OpenAlex. Falls back to "
                             "$OPENALEX_MAILTO (then $CROSSREF_MAILTO).")
    parser.add_argument("--core_api_key", default=None,
                        help="CORE API key. Falls back to $CORE_API_KEY.")
    parser.add_argument("--refresh_downloads", action="store_true",
                        help="Clear the round's downloads/ folder before fetching.")
    return parser

def main():
    args = _make_parser().parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        raise FileNotFoundError(f"queries.json not found: {queries_path}")

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        round_dir = Path(f"runs/{args.domain}/test_expand_{ts}")
    else:
        round_dir = Path(args.output)
    round_dir.mkdir(parents=True, exist_ok=True)
    print(f"[expand_round] Using round dir: {round_dir}")

    # Domain + last_tier
    spec = get_spec(args.domain)
    queries_preview = load_queries(str(queries_path))
    spec = _resolve_last_tier(args, spec, queries_preview)

    # Sources: --sources overrides; otherwise per-domain default.
    if args.sources:
        sources = parse_sources(args.sources)
    else:
        sources = spec.default_sources
    if not sources:
        raise RuntimeError(
            f"No usable sources for domain '{args.domain}'. "
            f"Pass --sources explicitly."
        )

    # CORE: if requested but no key, warn and drop it instead of crashing.
    core_api_key = args.core_api_key or os.getenv("CORE_API_KEY")
    if "core" in sources and not core_api_key:
        print("[expand_round] CORE_API_KEY missing — skipping CORE for this run.")
        sources = tuple(s for s in sources if s != "core")
        if not sources:
            raise RuntimeError(
                "CORE was the only requested source but CORE_API_KEY is missing. "
                "Either set CORE_API_KEY or pass --sources with another option."
            )

    # PubMed identity is optional but loud-encouraged.
    if "pubmed" in sources and not os.getenv("NCBI_EMAIL"):
        print("[expand_round] NCBI_EMAIL not set — PubMed will still run but "
              "NCBI may rate-limit aggressively. Consider setting NCBI_EMAIL "
              "and NCBI_API_KEY.")

    print(f"[expand_round] Domain={args.domain}  last_tier={spec.tiers[-1]!r}")
    print(f"[expand_round] Sources={sources}")

    # Refresh
    if args.refresh_downloads:
        dl_dir = round_dir / "downloads"
        if dl_dir.exists():
            for p in dl_dir.glob("*"):
                try:
                    p.unlink()
                except IsADirectoryError:
                    pass

    # Polite-pool emails (no school defaults).
    contact_emails: Dict[str, str] = {}
    if args.crossref_email:
        contact_emails["CROSSREF_MAILTO"] = args.crossref_email
    if args.openalex_email:
        contact_emails["OPENALEX_MAILTO"] = args.openalex_email

    result = run_one_round_search_and_download(
        queries_path=queries_path,
        round_dir=round_dir,
        domain_spec=spec,
        sources=sources,
        core_api_key=core_api_key,
        results_per_source=args.results_per_source,
        contact_emails=contact_emails or None,
    )

    pdfs = list(result["selected_pdf_paths"])
    print("\n[expand_round] Completed this round.")
    print(f" - Metadata        : {result['search_results_path']}")
    print(f" - Download dir    : {result['download_dir']}")
    print(f" - Downloaded meta : {result['downloaded_metadata_path']}")
    print(f" - Downloaded PDFs : {len(pdfs)}")
    for p in pdfs[:3]:
        print(f"   • {p.name}")
    if len(pdfs) > 3:
        print(f"   … (+{len(pdfs) - 3} more)")


if __name__ == "__main__":
    main()