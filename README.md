# AutoSchema Artifact

This repository contains the public camera-ready artifact for **AutoSchema: Self-Prompted Schema Induction and Evidence-Grounded Extraction**.

It includes:

1. **Phase I query generation** from seed PDFs.
2. **Phase I retrieval** from configurable academic sources.
3. **Phase I iterative orchestration** with deduplication, top-off, seed refresh, schema initialization/refinement, and final schema freezing.
4. **Phase I schema induction** with relevance filtering and semantic duplicate merge.
5. **Phase II evidence-grounded extraction** with schema-conditioned passage retrieval, multi-value extraction, JSONL audit logs, and flattened CSV output.

The public artifact intentionally removes ACS browser-based downloading because it depends on institutional/browser access and is not reliable for public reproduction. ACS, ACL, OpenReview, and Serper are not part of this artifact.

## Scope

| Domain | Last-tier setting | Artifact role |
|---|---|---|
| `chem` | `cof` / `mof` / `zif` | COF is the main benchmark; MOF/ZIF are chemistry transfer checks |
| `ad` | `amyloid` default | preliminary biomedical transfer check |
| `cs` | `prompt` default | query generation + retrieval only |

For `chem`, `--last_tier` is required. For `ad` and `cs`, it is optional.

CS schema induction and CS Phase II extraction are intentionally not part of this KDD artifact.

## Repository layout

```text
config/
  config.example.yaml
expand_lib/
prompt_lib/
schema_lib/
Phase2/
  extract_schema_values.py
  extract_lib/
expand_round.py
prompt_pipeline.py
design_extraction_prompt.py
iterative_expand.py
.env.example
.gitignore
requirements.txt
README.md
```

Generated outputs, downloaded PDFs, local configs, and real `.env` files should not be committed.

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

If you do not use `requirements.txt`, install the core dependencies manually:

```bash
pip install openai python-dotenv PyMuPDF pypdf PyYAML requests arxiv
```

Create local config files:

```bash
cp .env.example .env
cp config/config.example.yaml config/config.yaml   # optional non-secret defaults
```

Put API keys and real endpoints only in `.env`. Do not commit `.env` or `config/config.yaml`.

## Configuration

`prompt_lib/llm_client.py` resolves configuration in this order:

1. built-in defaults,
2. `config/config.yaml` if present,
3. environment variables / `.env`.

Main LLM variables:

| Env var | Meaning |
|---|---|
| `LLM_PROVIDER` | `azure_openai` or `openai` |
| `CHAT_DEPLOYMENT` | Azure chat deployment name |
| `CHAT_MODEL` | OpenAI model name when using `openai` |
| `CHAT_API_KEY` | chat API key; also accepted for OpenAI provider |
| `OPENAI_API_KEY` | OpenAI API key fallback when `LLM_PROVIDER=openai` |
| `CHAT_ENDPOINT` | Azure endpoint |
| `CHAT_VERSION` | Azure API version |
| `EMBEDDING_DEPLOYMENT` | Azure embedding deployment name |
| `EMBEDDING_MODEL` | OpenAI embedding model name |
| `EMBEDDING_API_KEY` | optional; falls back to `CHAT_API_KEY` |
| `EMBEDDING_ENDPOINT` | optional; falls back to `CHAT_ENDPOINT` |
| `EMBEDDING_VERSION` | Azure embedding API version |

Retrieval variables:

| Env var | Used by | Notes |
|---|---|---|
| `CORE_API_KEY` | CORE | if missing and CORE is requested, CORE is skipped with a warning |
| `CROSSREF_MAILTO` | Nature/Crossref | recommended polite-pool email |
| `OPENALEX_MAILTO` | OpenAlex | falls back to `CROSSREF_MAILTO` |
| `NCBI_EMAIL` | PubMed | recommended to reduce NCBI rate-limit issues |
| `NCBI_API_KEY` | PubMed | optional NCBI API key |

arXiv robustness variables:

| Env var | Meaning |
|---|---|
| `ARXIV_DISABLE=1` | skip arXiv for the whole run |
| `ARXIV_MIN_GAP` | minimum seconds between arXiv API calls; default `5.0` |
| `ARXIV_MAX_PAGE` | page-size cap for arXiv calls; default `25` |
| `ARXIV_FAIL_RETRIES` | quick retry count after arXiv 429/503 |
| `ARXIV_429_TRIP` | consecutive arXiv 429/503 failures before circuit breaker disables arXiv for the process |

If arXiv is temporarily rate-limited, AutoSchema can continue with the remaining sources.

## Phase I: query generation

Each generated query contains:

```text
tiers / query_bool / query_nl / synonyms / why
```

Example commands:

```bash
python prompt_pipeline.py --domain chem --last_tier cof --seeds data/seeds_COF --output tests/chem/cof_queries
python prompt_pipeline.py --domain chem --last_tier mof --seeds data/seeds_MOF --output tests/chem/mof_queries
python prompt_pipeline.py --domain chem --last_tier zif --seeds data/seeds_ZIF --output tests/chem/zif_queries
python prompt_pipeline.py --domain ad --seeds data/seeds_ad --output tests/ad/queries
python prompt_pipeline.py --domain cs --seeds data/seeds_prompt --output tests/cs/queries
```

Outputs include:

```text
papers.jsonl
queries.json
seeds/
```

## Phase I: retrieval

`expand_round.py` reads `queries.json`, searches multiple sources, merges candidates, deduplicates by `DOI -> source ID -> normalized title`, prioritizes downloadable papers, writes `search_results.jsonl`, and downloads PDFs into `downloads/`.

Default retrieval sources:

| Domain setting | Default sources |
|---|---|
| `--domain chem --last_tier cof/mof/zif` | `arxiv,openalex,nature,core` |
| `--domain ad` | `pubmed,biorxiv,openalex,core` |
| `--domain cs` | `arxiv,openalex` |

Supported sources:

```text
arxiv, openalex, nature, core, pubmed, biorxiv
```

Example commands:

```bash
python expand_round.py --domain chem --last_tier cof --queries tests/chem/cof_queries/queries.json --output tests/chem/cof_retrieve
python expand_round.py --domain ad --queries tests/ad/queries/queries.json --output tests/ad/retrieve
python expand_round.py --domain cs --queries tests/cs/queries/queries.json --output tests/cs/retrieve
```

Use `--sources` to override defaults, for example:

```bash
python expand_round.py --domain chem --last_tier cof \
  --queries tests/chem/cof_queries/queries.json \
  --output tests/chem/cof_retrieve \
  --sources openalex,nature,core
```

## Phase I: schema induction

`design_extraction_prompt.py` reads retrieved PDFs and induces/refines a compact schema. It applies relevance filtering before schema design and performs a final semantic duplicate merge by default.

Supported schema-induction domains:

| Domain setting | Meaning |
|---|---|
| `--domain chem --last_tier cof/mof/zif` | chemistry schema induction |
| `--domain ad` | preliminary amyloid-focused AD schema induction |

Initialize a schema:

```bash
python design_extraction_prompt.py --domain chem --last_tier cof \
  --pdf_dir tests/chem/cof_retrieve/downloads \
  --output tests/chem/cof_schema_r1 \
  --max_papers 20 \
  --max_chars_per_paper 12000
```

Refine a schema in a later round:

```bash
python design_extraction_prompt.py --domain chem --last_tier cof \
  --pdf_dir tests/chem/cof_round2/downloads \
  --prev_schema tests/chem/cof_schema_r1/final_schema.json \
  --output tests/chem/cof_schema_r2 \
  --max_papers 20 \
  --max_chars_per_paper 12000
```

Schema induction outputs include:

```text
candidate_fields.json
final_schema.json
extraction_prompt.txt
paper_relevance.json
schema_debug.json
semantic_merge.json
```

For very small corpora, schema induction may produce a partial schema. A larger and more relevant retrieved corpus usually produces a more useful compact schema.

## Phase I: iterative orchestrator

`iterative_expand.py` connects query generation, retrieval, dedup/top-off, schema init/refine, seed refresh, and final schema freezing.

Example COF run:

```bash
python iterative_expand.py \
  --domain chem \
  --last_tier cof \
  --seed_dir data/seeds_COF \
  --output_root tests/chem/cof_iterative \
  --rounds 2 \
  --results_per_source 10 \
  --round_depth_increment 5 \
  --target_total_papers 60 \
  --max_schema_papers 18 \
  --max_chars_per_paper 12000 \
  --min_round_accept_ratio 0.8 \
  --topoff_overfetch_ratio 1.5 \
  --overlap_threshold 0.15 \
  --sources arxiv,openalex,nature,core
```

If arXiv is rate-limited, either rely on the circuit breaker or run with:

```bash
--sources openalex,nature,core
```

Important outputs:

```text
round_01/final_downloaded_metadata.jsonl
round_01/accepted_metadata.jsonl
round_01/schema/final_schema.json
round_01/seed_refresh_scores.json
...
cumulative_metadata.jsonl
final_schemas/final_schema.json
final_schemas/extraction_prompt.txt
```

## Phase II: evidence-grounded extraction

Phase II uses a frozen schema from Phase I. It performs schema-conditioned passage retrieval, then asks the LLM to extract zero, one, or multiple values per field with evidence.

Output format in `extractions.jsonl` preserves full multi-value records and evidence. The CSV output flattens multi-values using ` | ` and writes missing values as `N/A`.

Example smoke test:

```bash
python Phase2/extract_schema_values.py \
  --pdf_dir data/phase2_pdfs \
  --schema_path runs/chem/cof_iterative/final_schemas/final_schema.json \
  --out_dir outputs/phase2_extractions \
  --max_papers 20 \
  --max_chars_per_paper 12000
```

If the corpus may contain retrieval noise, optionally rank papers by schema relevance before extraction:

```bash
python Phase2/extract_schema_values.py \
  --pdf_dir data/phase2_pdfs \
  --schema_path runs/chem/cof_iterative/final_schemas/final_schema.json \
  --out_dir outputs/phase2_extractions \
  --top_n 20 \
  --max_chars_per_paper 12000
```

Phase II outputs include:

```text
extractions.jsonl
schema_values.csv
paper_ranking.csv   # only when --top_n is used
```

## What not to commit

Do not commit:

```text
.env
config/config.yaml
downloaded PDFs
runs/
tests/
outputs/
output/
__pycache__/
.DS_Store
__MACOSX/
```

Downloaded publisher PDFs are intentionally excluded from the public artifact because of copyright and access restrictions. Users should provide their own seed PDFs and local corpora when running the artifact.

## Notes for reproducing paper-scale behavior

Exact schema fields and corpus composition may vary with seed PDFs, source availability, API rate limits, and retrieval noise. The artifact supports the same overall workflow, but public API availability may affect the exact number of downloaded PDFs in a given run.
