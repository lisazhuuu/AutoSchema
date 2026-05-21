import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ============================== Last-tier registry ==============================
@dataclass
class LastTier:
    key: str # Short CLI key (cof / mof / zif / amyloid / prompt)
    text: str # Full text appended to SPEC.tiers
    seeds_dir: str # Default seed PDFs folder
    aliases: List[str] # Used by validate_tiers

LAST_TIERS: Dict[str, LastTier] = {
    "cof": LastTier(
        key="cof",
        text="Covalent Organic Frameworks (COFs)",
        seeds_dir="data/seeds_cof",
        aliases=[
            "COF", "COFs",
            "covalent organic framework", "covalent-organic framework",
            "covalent organic frameworks",
        ],
    ),
    "mof": LastTier(
        key="mof",
        text="Metal-Organic Frameworks (MOFs)",
        seeds_dir="data/seeds_mof",
        aliases=[
            "MOF", "MOFs",
            "metal-organic framework", "metal organic framework",
            "metal-organic frameworks",
        ],
    ),
    "zif": LastTier(
        key="zif",
        text="Zeolitic Imidazolate Frameworks (ZIFs)",
        seeds_dir="data/seeds_zif",
        aliases=[
            "ZIF", "ZIFs",
            "zeolitic imidazolate framework",
            "zeolitic imidazolate frameworks",
        ],
    ),
    "amyloid": LastTier(
        key="amyloid",
        text="Amyloid production and APP processing",
        seeds_dir="data/seeds_ad",
        aliases=[
            "amyloid", "amyloid-beta", "amyloid beta", "Abeta", "A-beta",
            "APP", "APP processing",
            "amyloid precursor protein",
            "beta-secretase", "BACE1",
            "gamma-secretase",
        ],
    ),
    "prompt": LastTier(
        key="prompt",
        text="Prompt engineering for large language models",
        seeds_dir="data/seeds_cs",
        aliases=[
            "prompt engineering", "prompting",
            "LLM prompt", "LLM prompting",
            "large language model", "large language models",
        ],
    ),
}

# ============================== DomainSpec ==============================
@dataclass
class DomainSpec:
    domain_key: str
    base_tiers: List[str]
    arxiv_bool: str
    system_role: str
    domain_aspects: List[str]
    extract_prompt_template: str
    valid_last_tiers: List[str] # Subset of LAST_TIERS keys
    default_last_tier: Optional[str] = None  # None => --last_tier is REQUIRED
    required_terms: List[str] = field(default_factory=list)
    block_terms: List[str] = field(default_factory=list)

    # Resolved at runtime via .resolve(...)
    last_tier_key: str = ""
    last_tier_aliases: List[str] = field(default_factory=list)
    seeds_dir: str = ""

    # ---------------- Derived helpers ----------------
    @property
    def tiers(self) -> List[str]:
        if not self.last_tier_key:
            raise RuntimeError(
                f"SPEC for domain '{self.domain_key}' is unresolved; "
                f"call SPEC.resolve(last_tier_key) first."
            )
        return list(self.base_tiers) + [LAST_TIERS[self.last_tier_key].text]

    @property
    def last_tier(self) -> str:
        return self.tiers[-1] if self.last_tier_key else ""

    @property
    def extract_prompt(self) -> str:
        return self.extract_prompt_template

    @property
    def prompt_gen_domain(self) -> str:
        return _build_domain_prompt(self)

    @property
    def prompt_gen_general(self) -> str:
        return _build_general_prompt(self)

    # ---------------- Resolution ----------------
    def resolve(self, last_tier_key: Optional[str]) -> "DomainSpec":
        chosen = (last_tier_key or self.default_last_tier or "").lower() or None
        if not chosen:
            raise RuntimeError(
                f"Domain '{self.domain_key}' requires --last_tier "
                f"(choose from {self.valid_last_tiers})."
            )
        if chosen not in self.valid_last_tiers:
            raise RuntimeError(
                f"Domain '{self.domain_key}' does not support last_tier='{chosen}'. "
                f"Valid options: {self.valid_last_tiers}"
            )
        lt = LAST_TIERS[chosen]
        self.last_tier_key     = chosen
        self.last_tier_aliases = list(lt.aliases)
        self.seeds_dir         = lt.seeds_dir
        return self

# ============================== Prompt builders ==============================
def _aliases_str(spec: "DomainSpec") -> str:
    aliases = spec.last_tier_aliases or []
    return ", ".join(aliases) if aliases else spec.last_tier

def _aspects_str(spec: "DomainSpec") -> str:
    bullets = spec.domain_aspects or [
        "any relevant theme that appears in the seed papers"
    ]
    return "\n                - ".join(bullets)

def _build_domain_prompt(spec: DomainSpec) -> str:
    return f"""
        Return ONLY a JSON object: {{"queries":[...]}} with 8 items.

        Each item MUST contain exactly these keys:
            - "tiers": exactly {spec.tiers}
            - "query_bool": Boolean search query (arXiv-style)
            - "query_nl":   1-2 sentence natural-language description
            - "synonyms":   up to 3 related terms
            - "why":        purpose in <= 12 words

        Use the LAST item in the tiers list as the research focus:
            "{spec.last_tier}"
        All 8 queries MUST clearly target this last-tier topic as reflected by
        the seed papers (use the Extracted snapshot supplied below).

        Cover different aspects of this last-tier topic across the 8 queries,
        such as:
                - {_aspects_str(spec)}

        Boolean-query guidance:
            - Prefer arXiv categories: {spec.arxiv_bool}
            - Use ti: for titles and abs: for abstracts; combine with AND/OR
            - Each query MUST mention the last-tier keyword or one of its
              aliases ({_aliases_str(spec)}) in query_bool, query_nl, or
              synonyms.

        Use these EXACT tiers in every item. Do NOT invent new tiers or modify
        their text.

        Be concise.
    """

def _build_general_prompt(spec: DomainSpec) -> str:
    return f"""
        Return ONLY a JSON object: {{"queries":[...]}} with 10 items.

        Each item MUST contain exactly these keys:
            - "tiers": exactly {spec.tiers}
            - "query_bool": Boolean search query (arXiv-style)
            - "query_nl":   1-2 sentence natural-language description
            - "synonyms":   up to 3 related terms
            - "why":        purpose in <= 12 words

        Scope: {spec.system_role}. Focus queries on the LAST element of the
        tiers list:
            "{spec.last_tier}"

        Across the 10 queries, cover diverse aspects of this last-tier focus,
        such as:
                - {_aspects_str(spec)}

        Boolean-query guidance:
            - Prefer arXiv categories: {spec.arxiv_bool}
            - Use ti: for titles and abs: for abstracts; combine with AND/OR
            - Each query should mention the last-tier keyword or one of its
              aliases ({_aliases_str(spec)}).

        Use these EXACT tiers in every item.

        Be concise.
    """

# ============================== Extraction-prompt templates ==============================
CHEM_EXTRACT_TEMPLATE = """
    Return ONLY valid JSON. Use null or [] if unknown. No prose, no backticks.

    Top-level keys MUST be exactly:
        metadata, target_material, synthesis, properties, biomed, notes

    Schema:
        - metadata:        {title, authors, year, venue, doi, url}
        - target_material: {family, compound_name, formula}
        - synthesis:       {metal_source[], linker[], solvent[],
                            temperature_C[], time_h[], pH[], modulators[]}
        - properties:      {crystallinity, water_stability,
                            thermal_stability_C[],
                            gas_sorption: {NO, H2S, H2}}
        - biomed:          {toxicity_or_biocompatibility, degradation_trigger}
        - notes: []

    Text:
`````{paper_text}```
"""

CS_EXTRACT_TEMPLATE = """
    Return ONLY valid JSON. Use null or [] if unknown. No prose, no backticks.

    Top-level keys MUST be exactly:
        metadata, target_material, synthesis, properties, biomed, notes

    Schema (CS / prompt engineering):
        - metadata:        {title, authors, year, venue, doi, url}
        - target_material: {tasks[], benchmarks[], datasets[],
                            languages[], models[]}
        - synthesis:       {prompt_methods: {zero_shot, few_shot, cot,
                                self_consistency, self_refine,
                                tool_use_or_rag, demo_selection,
                                instruction_templates[], other[]},
                            auto_prompting: {present, search_space[],
                                objective[], optimizer[], iterations,
                                budget_tokens}}
        - properties:      {evaluation: {metrics[], baselines[], ablations[],
                                runs, variance, significance_test},
                            cost_and_latency: {tokens_in, tokens_out,
                                total_tokens, api_cost_usd, latency_ms}}
        - biomed:          {reproducibility: {code, data, seed_control,
                                artifact_links[]}}
        - notes: [key_findings[], failure_modes[]]

    Text:
````{paper_text}```
"""

AD_EXTRACT_TEMPLATE = """
    Return ONLY valid JSON. Use null or [] if unknown. No prose, no backticks.

    Top-level keys MUST be exactly:
        metadata, target_material, synthesis, properties, biomed, notes

    Schema (Alzheimer's disease):
        - metadata:        {title, authors, year, venue, doi, url}
        - target_material: {pathway, related_proteins[], related_genes[],
                            related_pathologies[]}
        - synthesis:       {model_systems[], cell_lines[], animal_models[],
                            assays[], reagents[]}
        - properties:      {quantitative_findings[], qualitative_findings[],
                            effect_size, statistical_test}
        - biomed:          {disease_relevance, therapeutic_target,
                            biomarker, clinical_stage}
        - notes: []

    Text:
```{paper_text}```
"""

# ============================== Domain registry ==============================
_BASE_CHEM_TIERS = [
    "Physical sciences",
    "Chemistry",
    "Materials chemistry",
    "Porous materials",
]

_BASE_CS_TIERS = [
    "Computer science",
    "Artificial intelligence",
    "Machine learning",
]

_BASE_AD_TIERS = [
    "Health sciences",
    "Neuroscience",
    "Neurodegenerative disease",
    "Alzheimer's disease",
]

_CHEM_DOMAIN_ASPECTS = [
    "how it is synthesized or prepared",
    "how its structure or composition is characterized",
    "how its stability or degradation is evaluated",
    "how its physical/chemical properties or performance are measured",
    "how it is applied in different contexts",
    "any other important themes that appear in the seed papers",
]

_CS_DOMAIN_ASPECTS = [
    "how it is formulated or proposed (problem setup, method design)",
    "how its components or design choices are characterized",
    "how its assumptions, limitations, or failure modes are evaluated",
    "how its performance is measured (datasets, benchmarks, metrics)",
    "how it is applied or deployed in different settings",
    "any other important themes that appear in the seed papers",
]

_AD_DOMAIN_ASPECTS = [
    "how it is studied or modeled (model systems, assays, cohorts)",
    "how its molecular or cellular components are characterized",
    "how its dysregulation, progression, or risk factors are evaluated",
    "how its biomarkers or quantitative effects are measured",
    "how it is targeted therapeutically or diagnostically in different contexts",
    "any other important themes that appear in the seed papers",
]

DOMAINS: Dict[str, DomainSpec] = {
    "chem": DomainSpec(
        domain_key="chem",
        base_tiers=_BASE_CHEM_TIERS,
        arxiv_bool="(cat:cond-mat.mtrl-sci OR cat:physics.chem-ph)",
        system_role="chemistry / materials",
        domain_aspects=_CHEM_DOMAIN_ASPECTS,
        extract_prompt_template=CHEM_EXTRACT_TEMPLATE,
        valid_last_tiers=["cof", "mof", "zif"],
        default_last_tier=None,                 # --last_tier is REQUIRED
    ),
    "cs": DomainSpec(
        domain_key="cs",
        base_tiers=_BASE_CS_TIERS,
        arxiv_bool="(cat:cs.CL OR cat:cs.AI OR cat:cs.LG OR cat:cs.SE)",
        system_role="CS prompt engineering",
        domain_aspects=_CS_DOMAIN_ASPECTS,
        extract_prompt_template=CS_EXTRACT_TEMPLATE,
        valid_last_tiers=["prompt"],
        default_last_tier="prompt",
    ),
    "ad": DomainSpec(
        domain_key="ad",
        base_tiers=_BASE_AD_TIERS,
        arxiv_bool="(cat:q-bio.NC OR cat:q-bio.BM OR cat:q-bio.MN OR cat:q-bio.SC)",
        system_role="Alzheimer's disease / neuroscience",
        domain_aspects=_AD_DOMAIN_ASPECTS,
        extract_prompt_template=AD_EXTRACT_TEMPLATE,
        valid_last_tiers=["amyloid"],
        default_last_tier="amyloid",
    ),
}

# ============================== CLI parsing ==============================
def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--domain", choices=list(DOMAINS.keys()))
    parser.add_argument(
        "--last_tier", "--last-tier",
        dest="last_tier", type=str, default=None,
        help="Last-tier key: one of cof / mof / zif / amyloid / prompt.",
    )
    args, _ = parser.parse_known_args()
    return args

def get_domain() -> str:
    env = os.getenv("DOMAIN")
    cli = _parse_cli()
    chosen = (cli.domain or env or "chem").lower()
    if chosen not in DOMAINS:
        raise ValueError(
            f"Unknown domain '{chosen}'. Available: {sorted(DOMAINS.keys())}"
        )
    return chosen

def get_last_tier_key() -> Optional[str]:
    env = os.getenv("LAST_TIER")
    cli = _parse_cli()
    val = cli.last_tier or env
    return val.lower() if val else None

# ============================== Module-level resolved SPEC ==============================
DOMAIN = get_domain()
SPEC   = DOMAINS[DOMAIN].resolve(get_last_tier_key())

DATA_DIR = Path(SPEC.seeds_dir)
_OUTPUT_ROOT = os.getenv("AUTOSCHEMA_OUTPUT_DIR", "output")
OUT_DIR = Path(_OUTPUT_ROOT) / DOMAIN / SPEC.last_tier_key
OUT_DIR.mkdir(parents=True, exist_ok=True)