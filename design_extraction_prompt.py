from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schema_lib.domain   import SCHEMA_DOMAINS, resolve_domain_and_last_tier
from schema_lib.io_utils import save_json, save_text, load_json
from schema_lib.sampling import load_pdf_texts, relevance_filtered_sample
from schema_lib.fields   import (
    propose_candidate_fields_per_paper,
    merge_candidate_fields,
    select_final_schema,
)
from schema_lib.refine   import refine_schema
from schema_lib.merge    import semantic_merge_fields
from schema_lib.prompt   import build_extraction_prompt

# CLI
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AutoSchema Phase-I schema induction (KDD camera-ready artifact)."
    )
    p.add_argument(
        "--domain", required=True,
        choices=list(SCHEMA_DOMAINS.keys()),
        help="Schema-induction domain. KDD artifact supports: chem | ad.",
    )
    p.add_argument(
        "--last_tier", "--last-tier",
        dest="last_tier", default=None,
        choices=["cof", "mof", "zif", "amyloid"],
        help="Retrieval-focus last tier. "
             "REQUIRED for --domain chem (cof|mof|zif). "
             "Defaults to 'amyloid' for --domain ad.",
    )
    p.add_argument(
        "--pdf_dir", required=True,
        help="Folder of retrieved PDFs (e.g. runs/chem/cof_retrieve/downloads).",
    )
    p.add_argument(
        "--output", "--out_dir",
        dest="output", default=None,
        help="Output directory. Defaults to runs/<domain>/<last_tier>/"
             "schema_<timestamp>.",
    )
    p.add_argument(
        "--prev_schema", default=None,
        help="(Optional) Path to a previous final_schema.json. If provided, "
             "the script REFINES that schema using fields induced from the "
             "new PDFs instead of starting from scratch.",
    )
    p.add_argument("--max_papers", type=int, default=20,
                   help="Max number of PDFs used for schema induction "
                        "(sampled from the top-75% relevance pool).")
    p.add_argument("--relevance_drop_fraction", type=float, default=0.25,
                   help="Fraction of least-relevant papers to drop "
                        "(default 0.25 = bottom 25%%).")
    p.add_argument("--relevance_chars", type=int, default=4000,
                   help="Chars of each paper used for embedding-based "
                        "relevance scoring.")
    p.add_argument("--max_chars_per_paper", type=int, default=12000,
                   help="Chars of each paper fed to the LLM for per-paper "
                        "candidate-field proposal.")
    p.add_argument("--extra_keywords", default="",
                   help="Optional comma-separated keywords appended to the "
                        "last-tier text when computing relevance scores.")
    p.add_argument("--max_fields_per_paper", type=int, default=15)
    p.add_argument("--max_final_fields", type=int, default=12)
    p.add_argument(
        "--min_support", type=int, default=2,
        help="Minimum number of sampled papers a NEW field must appear in. "
             "Auto-relaxed to 1 when fewer than 2 PDFs were sampled. "
             "Previous fields are not enforced against this threshold "
             "during refinement.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_semantic_merge", action="store_true",
                   help="Disable the final LLM semantic-duplicate merge pass "
                        "(on by default). When enabled, merges duplicate "
                        "fields, keeps one per concept, and drops overly broad "
                        "fields before the schema is frozen.")
    p.add_argument("--no_debug", action="store_true",
                   help="If set, do not write schema_debug.json, "
                        "paper_relevance.json, or semantic_merge.json.")
    return p

def main():
    args = _make_parser().parse_args()
    spec, last_tier_key, last_tier_text = resolve_domain_and_last_tier(args)

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    if args.output:
        out_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"runs/{args.domain}/{last_tier_key}/schema_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "refine" if args.prev_schema else "initialize"
    print(f"[schema] mode={mode}  domain={args.domain}  "
          f"last_tier={last_tier_key}  text={last_tier_text!r}")
    print(f"[schema] pdf_dir={pdf_dir}")
    print(f"[schema] out_dir={out_dir}")

    # ---------- Load all PDFs and relevance-filter ----------
    all_papers = load_pdf_texts(
        pdf_dir,
        relevance_chars=args.relevance_chars,
        proposal_chars=args.max_chars_per_paper,
    )
    extra_kws = [k.strip() for k in args.extra_keywords.split(",") if k.strip()]
    selected, relevance_debug = relevance_filtered_sample(
        papers=all_papers,
        last_tier_text=last_tier_text,
        extra_keywords=extra_kws,
        max_papers=args.max_papers,
        drop_fraction=args.relevance_drop_fraction,
        seed=args.seed,
    )
    if not args.no_debug:
        save_json(out_dir / "paper_relevance.json", relevance_debug)

    if not selected:
        raise SystemExit("[schema] no papers survived relevance filtering.")

    effective_min_support = args.min_support
    if len(selected) < 2 and effective_min_support > 1:
        print(f"[schema] only {len(selected)} sampled paper(s) -> "
              f"auto-relaxing min_support {args.min_support} -> 1")
        effective_min_support = 1

    # ---------- Per-paper candidate-field proposal ----------
    per_paper = propose_candidate_fields_per_paper(
        papers=selected,
        spec=spec,
        last_tier_text=last_tier_text,
        max_fields_per_paper=args.max_fields_per_paper,
    )

    # ---------- Canonical merge ----------
    canonicals = merge_candidate_fields(
        per_paper=per_paper,
        spec=spec,
        last_tier_text=last_tier_text,
    )
    canonicals_for_disk = [
        {k: v for k, v in c.items() if k != "_signals"}
        for c in canonicals
    ]
    save_json(out_dir / "candidate_fields.json", canonicals_for_disk)

    # ---------- Initial selection OR iterative refinement ----------
    if args.prev_schema:
        prev_path = Path(args.prev_schema)
        if not prev_path.exists():
            raise SystemExit(f"--prev_schema not found: {prev_path}")
        previous_fields = load_json(prev_path)
        if not isinstance(previous_fields, list):
            raise SystemExit(
                f"--prev_schema must be a JSON list of fields; got "
                f"{type(previous_fields).__name__}"
            )
        print(f"[schema] refining over {len(previous_fields)} previous fields "
              f"from {prev_path}")
        final_fields, debug_rows = refine_schema(
            previous_fields=previous_fields,
            canonicals=canonicals,
            spec=spec,
            last_tier_text=last_tier_text,
            max_final_fields=args.max_final_fields,
            min_support=effective_min_support,
        )
    else:
        final_fields, debug_rows = select_final_schema(
            canonicals=canonicals,
            max_final_fields=args.max_final_fields,
            min_support=effective_min_support,
        )

    # ---------- Final semantic-duplicate merge (LLM, before freeze) ----------
    # Runs after select/refine so it cleans up BOTH the initialize and refine
    # paths. The LLM only groups/drops EXISTING fields; records are rebuilt in
    # code, and on any failure we keep the un-merged schema.
    merge_debug = {"applied": False, "before": len(final_fields),
                   "after": len(final_fields), "reason": "disabled"}
    if not args.no_semantic_merge:
        final_fields, merge_debug = semantic_merge_fields(
            final_fields=final_fields,
            spec=spec,
            last_tier_text=last_tier_text,
            max_final_fields=args.max_final_fields,
        )
        print(f"[schema] semantic merge: applied={merge_debug.get('applied')}  "
              f"{merge_debug.get('before')} -> {merge_debug.get('after')} fields")
        if merge_debug.get("groups_applied"):
            for g in merge_debug["groups_applied"]:
                print(f"  [merge] keep '{g['keep']}' <- {g['merge_from']}")
        if merge_debug.get("dropped"):
            print(f"  [merge] dropped (too broad): {merge_debug['dropped']}")

    save_json(out_dir / "final_schema.json", final_fields)
    if not args.no_debug:
        save_json(out_dir / "semantic_merge.json", merge_debug)

    # ---------- Extraction prompt ----------
    extraction_prompt = build_extraction_prompt(
        final_fields=final_fields,
        spec=spec,
        last_tier_text=last_tier_text,
    )
    save_text(out_dir / "extraction_prompt.txt", extraction_prompt)

    # ---------- Debug ----------
    if not args.no_debug:
        save_json(out_dir / "schema_debug.json", {
            "mode":                  mode,
            "domain":                args.domain,
            "last_tier_key":         last_tier_key,
            "last_tier_text":        last_tier_text,
            "extra_keywords":        extra_kws,
            "num_input_pdfs":        len(all_papers),
            "num_sampled_papers":    len(selected),
            "candidate_count":       len(canonicals),
            "final_count":           len(final_fields),
            "effective_min_support": effective_min_support,
            "scored_candidates":     debug_rows,
        })

    print("\n[schema] Wrote:")
    print(f"  - {out_dir / 'candidate_fields.json'}  ({len(canonicals)} canonical candidates)")
    print(f"  - {out_dir / 'final_schema.json'}      ({len(final_fields)} fields)")
    print(f"  - {out_dir / 'extraction_prompt.txt'}")
    if not args.no_debug:
        print(f"  - {out_dir / 'paper_relevance.json'}")
        print(f"  - {out_dir / 'schema_debug.json'}")


if __name__ == "__main__":
    main()
