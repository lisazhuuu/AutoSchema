from __future__ import annotations

import argparse, argparse, csv, sys
from pathlib import Path
from typing import List

# Make sibling extract_lib importable when run as `python extract_schema_values.py`
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from extract_lib.config import (
    DEFAULT_MAX_CHARS_PER_PAPER,
    DEFAULT_MODEL_DEPLOYMENT,
    DEFAULT_OUT_DIR,
    DEFAULT_PDF_DIR,
    DEFAULT_SCHEMA_PATH,
    N_PAPERS,
    SKIP_IF_DONE,
)
from extract_lib.llm_extract import build_chat_client, call_extract, get_model_deployment
from extract_lib.pdf_io import extract_pdf_text
from extract_lib.postprocess import normalize_extraction
from extract_lib.prompt_builder import build_system_prompt, build_user_prompt
from extract_lib.retrieval import (
    paper_relevance_signal,
    render_context_pack,
    retrieve_per_field_contexts,
)
from extract_lib.schema_loader import load_schema
from extract_lib.writer import (
    append_jsonl,
    load_all_records,
    load_done_paper_ids,
    rewrite_csv,
)

# CLI
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 schema-value extraction.")
    p.add_argument("--pdf_dir", "--pdf-dir", dest="pdf_dir", default=DEFAULT_PDF_DIR,
                   help="Directory containing the corpus PDFs.")
    p.add_argument("--schema_path", "--schema", dest="schema_path",
                   default=DEFAULT_SCHEMA_PATH,
                   help="Path to Phase-1 final_schema.json.")
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR,
                   help="Where to write extractions.jsonl + schema_values.csv.")
    p.add_argument("--prompt_path", "--prompt-path", dest="prompt_path", default=None,
                   help="(Optional) Path to a text file whose contents OVERRIDE "
                        "the system prompt. By default the system prompt is built "
                        "from the schema fields.")
    p.add_argument("--top_n", "--top-n", dest="top_n", type=int, default=None,
                   help="If set: rank ALL PDFs in --pdf_dir by relevance to the "
                        "schema (alias-keyword scoring), then keep only the top N. "
                        "Off-topic papers fall to the bottom and get skipped. "
                        "Recommended when the corpus may contain noise from Phase 1.")
    p.add_argument("--max_papers", "--n-papers", "--n_papers", dest="max_papers",
                   type=int, default=N_PAPERS,
                   help="(Alphabetical) When --top_n is NOT set, take the first N "
                        "PDFs in alphabetical order. Use -1 for all.")
    p.add_argument("--max_chars_per_paper", "--max-chars-per-paper",
                   dest="max_chars_per_paper", type=int,
                   default=DEFAULT_MAX_CHARS_PER_PAPER,
                   help="Truncate each paper's text to this many chars before "
                        "passage retrieval. 0 = no cap (default).")
    p.add_argument("--model", default=None,
                   help="Override the chat model / deployment name "
                        f"(default: repo config CHAT_DEPLOYMENT / CHAT_MODEL, "
                        f"else '{DEFAULT_MODEL_DEPLOYMENT}').")
    p.add_argument("--no-resume", "--no_resume", dest="no_resume",
                   action="store_true",
                   help="Disable skip-if-done behavior; overwrite outputs.")
    return p.parse_args()

# Relevance-ranking pass
def _rank_and_select(all_pdfs: List[Path], fields, top_n: int, out_dir: Path) -> List[Path]:
    print(f"📊 Ranking {len(all_pdfs)} PDFs by relevance "
          f"(this is a CPU-only pass, no LLM tokens spent)...")
    scored: List[tuple] = []   # (n_covered, score, n_paragraphs, pdf_path)
    for i, pdf in enumerate(all_pdfs, 1):
        text = extract_pdf_text(pdf)
        if not text or len(text) < 200:
            scored.append((0, 0.0, 0, pdf))
        else:
            rel = paper_relevance_signal(text, fields)
            scored.append((rel.n_fields_covered, rel.total_best_score, rel.n_paragraphs, pdf))
        if i % 20 == 0 or i == len(all_pdfs):
            print(f"   ranked {i}/{len(all_pdfs)}")

    # Sort: most fields covered first, then by total best score
    scored.sort(key=lambda r: (-r[0], -r[1]))

    # Write the full ranking so the user can audit what got kept/dropped
    ranking_path = out_dir / "paper_ranking.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ranking_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "selected", "n_fields_covered", "total_best_score",
                    "n_paragraphs", "paper_id"])
        for r, (ncov, sc, npara, pdf) in enumerate(scored, 1):
            w.writerow([
                r,
                "yes" if r <= top_n else "no",
                ncov,
                f"{sc:.2f}",
                npara,
                pdf.stem,
            ])

    kept = [r[3] for r in scored[:top_n]]
    print(f"🎯 Keeping top {len(kept)} / {len(all_pdfs)} PDFs by relevance "
          f"(see {ranking_path.name}).")
    # Surface a quick sanity preview
    if scored:
        top1 = scored[0]
        bot = scored[min(top_n, len(scored)) - 1] if top_n <= len(scored) else scored[-1]
        cut = scored[top_n] if top_n < len(scored) else None
        print(f"   ↑ best:      n_fields={top1[0]} score={top1[1]:.1f}  {top1[3].stem[:60]}")
        print(f"   = cutoff:    n_fields={bot[0]} score={bot[1]:.1f}  {bot[3].stem[:60]}")
        if cut is not None:
            print(f"   ↓ just out:  n_fields={cut[0]} score={cut[1]:.1f}  {cut[3].stem[:60]}")
    return kept

# Main
def main() -> int:
    args = _parse_args()

    pdf_dir = Path(args.pdf_dir)
    schema_path = Path(args.schema_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        return 1
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return 1

    # --- load schema ---
    fields = load_schema(schema_path)
    print(f"📐 Loaded {len(fields)} schema fields from {schema_path.name}")
    for f in fields:
        print(f"     • {f.name}  ({f.value_type}, {len(f.aliases)} aliases)")

    # System prompt: built from the schema, or overridden by --prompt_path.
    if args.prompt_path:
        prompt_file = Path(args.prompt_path)
        if not prompt_file.exists():
            print(f"❌ --prompt_path not found: {prompt_file}")
            return 1
        system_prompt = prompt_file.read_text(encoding="utf-8")
        print(f"📝 Using system prompt override from {prompt_file}")
    else:
        system_prompt = build_system_prompt(fields)

    # --- LLM client (provider-agnostic; credentials validated here, lazily) ---
    client = build_chat_client()
    model = args.model or get_model_deployment()
    print(f"🤖 Using chat model/deployment: {model}")

    # --- corpus (with optional relevance ranking) ---
    all_pdfs: List[Path] = sorted(pdf_dir.glob("*.pdf"))
    if not all_pdfs:
        print(f"❌ No PDFs found in {pdf_dir}")
        return 1

    if args.top_n is not None and args.top_n > 0:
        pdfs = _rank_and_select(all_pdfs, fields, args.top_n, out_dir)
    else:
        cap = args.max_papers
        if cap is not None and cap >= 0:
            pdfs = all_pdfs[:cap]
        else:
            pdfs = all_pdfs
        print(f"📚 (alphabetical mode) Processing {len(pdfs)} / {len(all_pdfs)} PDFs.")

    # --- output paths + resume ---
    jsonl_path = out_dir / "extractions.jsonl"
    csv_path = out_dir / "schema_values.csv"
    skipped_path = out_dir / "skipped_papers.txt"

    if args.no_resume:
        for p in (jsonl_path, csv_path, skipped_path):
            if p.exists():
                p.unlink()
        done_ids = set()
    else:
        done_ids = load_done_paper_ids(jsonl_path) if SKIP_IF_DONE else set()
    if done_ids:
        print(f"⏭️  Resume: {len(done_ids)} papers already in extractions.jsonl")

    # --- process loop ---
    skipped: List[str] = []
    n_ok = 0
    for i, pdf_path in enumerate(pdfs, 1):
        paper_id = pdf_path.stem
        if paper_id in done_ids:
            print(f"[{i}/{len(pdfs)}] ⏭  skip (done): {paper_id}")
            continue

        print(f"[{i}/{len(pdfs)}] 📄 {paper_id}")

        # 1) PDF -> text (optionally capped by --max_chars_per_paper)
        full_text = extract_pdf_text(pdf_path)
        if args.max_chars_per_paper and args.max_chars_per_paper > 0:
            full_text = full_text[: args.max_chars_per_paper]
        if not full_text or len(full_text) < 200:
            print("   ⚠️  PDF text too short — skipping.")
            skipped.append(paper_id)
            continue

        # 2) Retrieval: build the per-field evidence pack
        ctx = retrieve_per_field_contexts(full_text, fields)
        if not ctx.selected_indices:
            print("   ⚠️  No relevant passages found — skipping.")
            skipped.append(paper_id)
            continue
        context_pack = render_context_pack(ctx)
        print(f"   📎 Sending {len(ctx.selected_indices)} passages "
              f"(~{len(context_pack):,} chars) to the LLM.")

        # 3) LLM call
        user_prompt = build_user_prompt(paper_id, context_pack)
        raw = call_extract(client, model, system_prompt, user_prompt, paper_id)
        if raw is None:
            skipped.append(paper_id)
            continue

        # 4) Normalize + persist
        record = normalize_extraction(raw, fields)
        append_jsonl(jsonl_path, record)
        rewrite_csv(csv_path, fields, load_all_records(jsonl_path))
        n_ok += 1

        # quick summary so progress is visible
        non_empty = sum(
            1 for fname in (f.name for f in fields)
            if record["extractions"].get(fname, {}).get("values")
        )
        print(f"   ✅ extracted values for {non_empty}/{len(fields)} fields.")

    # --- write skipped list ---
    if skipped:
        skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")

    print()
    print(f"✅ done: {n_ok} successful, {len(skipped)} skipped.")
    print(f"   jsonl  → {jsonl_path}")
    print(f"   csv    → {csv_path}")
    if skipped:
        print(f"   skipped→ {skipped_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
