import json, uuid, argparse, time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

# Load a local .env if python-dotenv is installed (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Internal helpers
from prompt_lib.domain import SPEC, DOMAIN
from prompt_lib.validation import validate_tiers
from prompt_lib.paper_model import Paper, pdf_to_text
from prompt_lib.llm_client import llm_chat, get_config

# ================== arXiv-style category filter ==================
def inject_arxiv_cats(qb: str) -> str:
    qb = (qb or "").strip()
    cats = SPEC.arxiv_bool
    if not cats or "cat:" in qb.lower():
        return qb
    return f"({qb}) AND {cats}" if qb else cats

# ================== Metadata extraction helper ==================
def _meta_field(extracted: Dict[str, Any], key: str, *legacy_keys: str):
    if not isinstance(extracted, dict):
        return None
    meta = extracted.get("metadata")
    if isinstance(meta, dict) and meta.get(key) is not None:
        return meta.get(key)
    # Back-compat: legacy top-level keys (e.g. "doi_or_id" instead of "doi").
    for lk in (key,) + legacy_keys:
        if extracted.get(lk) is not None:
            return extracted.get(lk)
    return None

# ================== Main builder ==================
def build_queries_from_seed_pdfs(
    seed_pdfs: List[Path],
    round_dir: Path,
    max_queries: int = 8,
) -> Path:
    round_dir.mkdir(parents=True, exist_ok=True)

    seeds_dir = round_dir / "seeds"
    if seeds_dir.exists():
        for old_file in seeds_dir.glob("*"):
            try:
                old_file.unlink()
            except IsADirectoryError:
                pass
    else:
        seeds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract structured snapshots from seed PDFs
    papers: List[Paper] = []
    for pdf in seed_pdfs[:5]:
        dest = seeds_dir / pdf.name
        if not dest.exists():
            dest.write_bytes(pdf.read_bytes())

        txt_path = pdf_to_text(pdf)
        text = Path(txt_path).read_text(encoding="utf-8")

        extracted_raw = llm_chat(
            [
                {"role": "system",
                 "content": f"You are a {SPEC.system_role} extraction assistant. Return ONLY valid JSON."},
                {"role": "user",
                 "content": SPEC.extract_prompt.replace("{paper_text}", text)},
            ],
            force_json=True,
        )
        time.sleep(1)

        try:
            extracted = json.loads(extracted_raw)
        except Exception:
            extracted = {"_raw": extracted_raw}

        papers.append(Paper(
            paper_id=f"seed_{uuid.uuid4().hex[:8]}",
            title=_meta_field(extracted, "title"),
            authors=_meta_field(extracted, "authors"),
            year=_meta_field(extracted, "year"),
            venue=_meta_field(extracted, "venue"),
            doi=_meta_field(extracted, "doi", "doi_or_id"),
            source="seed",
            raw_text_path=txt_path,
            extracted=extracted,
        ))

    # Persist papers.jsonl
    with open(round_dir / "papers.jsonl", "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    snapshot = json.dumps([p.extracted for p in papers])[:8000]

    # 2. Domain prompt -> 8 queries; top up with general prompt if needed
    def run_query_prompt(prompt_text: str):
        raw = llm_chat(
            [
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user",
                 "content": prompt_text + f"\n\nExtracted snapshot:\n{snapshot}"},
            ],
            force_json=True,
        )
        time.sleep(1)
        try:
            obj = json.loads(raw)
            return obj.get("queries", [])
        except Exception:
            return []

    def _ingest(items, seen, merged):
        for item in items:
            if SPEC.tiers and "tiers" not in item:
                item["tiers"] = list(SPEC.tiers)

            qb_raw   = item.get("query_bool") or item.get("query_nl") or ""
            qb_final = inject_arxiv_cats(qb_raw)

            if not validate_tiers(item, qb_final, DOMAIN, SPEC):
                continue

            key = " ".join((qb_final or "").lower().split())
            if not key or key in seen:
                continue
            seen.add(key)

            item["query_bool"] = qb_final
            merged.append(item)
            if len(merged) >= max_queries:
                return True
        return False

    seen, merged = set(), []
    q_domain = run_query_prompt(SPEC.prompt_gen_domain)
    _ingest(q_domain, seen, merged)

    if len(merged) < max_queries:
        print(f"[prompt_pipeline] Domain prompt yielded {len(merged)}/{max_queries}; "
              f"falling back to prompt_gen_general.")
        q_general = run_query_prompt(SPEC.prompt_gen_general)
        _ingest(q_general, seen, merged)

    merged = merged[:max_queries]

    # 3. Write queries.json
    queries_path = round_dir / "queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    return queries_path

# ================== CLI ==================
def main():
    parser = argparse.ArgumentParser(
        description="Generate per-round queries.json from a set of seed PDFs."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Folder containing seed PDFs. Defaults to SPEC.seeds_dir for the chosen --domain.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["chem", "cs", "ad"],
        help="Domain key (chem | cs | ad). Consumed by prompt_lib.domain at import.",
    )
    parser.add_argument(
        "--last_tier", "--last-tier",
        dest="last_tier",
        type=str,
        default=None,
        help="Last-tier key: cof | mof | zif (required for --domain chem); "
             "amyloid (default for --domain ad); prompt (default for --domain cs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: runs/<domain>/test_prompts_<timestamp>).",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=8,
        help="Max number of unique search queries to keep in queries.json (paper default = 8).",
    )
    args = parser.parse_args()

    # Resolve seed dir: explicit --seeds wins, otherwise fall back to SPEC.seeds_dir.
    seed_dir = Path(args.seeds) if args.seeds else Path(SPEC.seeds_dir)
    if not seed_dir.exists():
        raise RuntimeError(
            f"Seed directory '{seed_dir}' does not exist. Pass --seeds or create "
            f"SPEC.seeds_dir for domain '{DOMAIN}'."
        )

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        round_dir = Path(f"runs/{DOMAIN}/{SPEC.last_tier_key}/test_prompts_{ts}")
    else:
        round_dir = Path(args.output)

    seed_pdfs = sorted(seed_dir.glob("*.pdf"))
    if not seed_pdfs:
        raise RuntimeError(f"No PDFs found in {seed_dir}")

    cfg = get_config()
    print(
        f"[prompt_pipeline] Provider: {cfg.provider}  "
        f"Domain: {DOMAIN}  Last tier: {SPEC.tiers[-1]!r}"
    )
    print(f"[prompt_pipeline] using {len(seed_pdfs)} seed PDFs from {seed_dir}")
    print(f"[prompt_pipeline] writing round outputs to {round_dir}")

    queries_path = build_queries_from_seed_pdfs(
        seed_pdfs=seed_pdfs,
        round_dir=round_dir,
        max_queries=args.max_queries,
    )

    print("\n[prompt_pipeline] Wrote:")
    print(f" - {round_dir / 'papers.jsonl'}")
    print(f" - {queries_path}")
    print(f" - Copies of seed PDFs under {round_dir / 'seeds'}")

if __name__ == "__main__":
    main()
