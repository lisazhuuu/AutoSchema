# ----- Default I/O paths -----
DEFAULT_PDF_DIR = "data/phase2_pdfs"
DEFAULT_SCHEMA_PATH = "runs/chem/cof_iterative/final_schemas/final_schema.json"
DEFAULT_OUT_DIR = "outputs/phase2_extractions"

# ----- Per-paper text cap -----
DEFAULT_MAX_CHARS_PER_PAPER = 0

# ----- Corpus iteration -----
N_PAPERS = 200                # cap on how many PDFs to process; set to None for all
SKIP_IF_DONE = True           # resume mode: skip paper_ids already present in extractions.jsonl

# ----- Retrieval (per-field paragraph selection) -----
PARAGRAPH_MIN_CHARS = 80      # drop very short fragments (headers, page numbers)
PARAGRAPH_MAX_CHARS = 1800    # split overly long paragraphs into windows of this size
TOP_K_PER_FIELD = 6           # how many top-scoring paragraphs to keep per schema field
GLOBAL_PARAGRAPH_BUDGET = 60  # hard cap on total distinct paragraphs sent to the LLM
ALIAS_HIT_WEIGHT = 2.0        # score multiplier for alias keyword hits
NAME_HIT_WEIGHT = 3.0         # extra boost when the canonical field name appears
EVIDENCE_HIT_WEIGHT = 0.5     # small boost from overlap with phase-1 evidence snippets

# ----- LLM -----
DEFAULT_MODEL_DEPLOYMENT = "gpt-4o"
MAX_COMPLETION_TOKENS = 4000  # array values + evidence per field need room
LLM_RETRIES = 3
LLM_RETRY_BACKOFF_SEC = 2.0
LLM_TEMPERATURE = 0.0         # deterministic extraction

# ----- Output formatting -----
MULTIVALUE_JOINER = " | "     # CSV cell joiner for array-typed fields
NA_VALUE = "N/A"
