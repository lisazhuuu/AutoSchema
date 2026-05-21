import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ================== DomainSpec (retrieval side) ==================
@dataclass
class DomainSpec:
    arxiv_cats: List[str]
    base_tiers: List[str]
    default_sources: Tuple[str, ...]
    valid_last_tiers: List[str]                       # subset of LAST_TIERS keys
    default_last_tier: Optional[str] = None           # None => --last_tier REQUIRED
    last_tier_key: str = ""                           # set by resolve()
    last_tier_aliases: List[str] = field(default_factory=list)

    @property
    def tiers(self) -> List[str]:
        if not self.last_tier_key:
            raise RuntimeError(
                "DomainSpec is unresolved; call spec.resolve(last_tier_key) first."
            )
        return list(self.base_tiers) + [LAST_TIERS[self.last_tier_key].text]

    def resolve(self, last_tier_key: Optional[str]) -> "DomainSpec":
        from dataclasses import replace
        chosen = (last_tier_key or self.default_last_tier or "").lower() or None
        if not chosen:
            raise RuntimeError(
                f"--last_tier is required for this domain. "
                f"Valid options: {self.valid_last_tiers}"
            )
        if chosen not in self.valid_last_tiers:
            raise RuntimeError(
                f"--last_tier '{chosen}' is not valid for this domain. "
                f"Valid options: {self.valid_last_tiers}"
            )
        lt = LAST_TIERS[chosen]
        return replace(self, last_tier_key=chosen,
                       last_tier_aliases=list(lt.aliases))

# Available retrieval sources
SEARCH_SOURCES: Dict[str, str] = {
    "arxiv":    "arXiv preprints",
    "openalex": "OpenAlex open-access metadata + PDFs",
    "nature":   "Crossref / Springer Nature member 297",
    "core":     "CORE open-access aggregator (requires CORE_API_KEY)",
    "pubmed":   "PubMed / PMC (NCBI eutils; uses NCBI_EMAIL / NCBI_API_KEY when set)",
    "biorxiv":  "bioRxiv biology preprints",
}

# Base tiers, arxiv categories, default sources
_CHEM_ARXIV_CATS = ["cond-mat.mtrl-sci", "physics.chem-ph"]
_CS_ARXIV_CATS   = ["cs.CL", "cs.AI", "cs.LG", "cs.SE"]
_AD_ARXIV_CATS   = ["q-bio.NC", "q-bio.BM", "q-bio.MN", "q-bio.SC"]

_CHEM_BASE_TIERS = ["Physical sciences", "Chemistry",
                    "Materials chemistry", "Porous materials"]
_CS_BASE_TIERS   = ["Computer science", "Artificial intelligence",
                    "Machine learning"]
_AD_BASE_TIERS   = ["Health sciences", "Neuroscience",
                    "Neurodegenerative disease", "Alzheimer's disease"]

@dataclass
class LastTier:
    key: str               # short CLI key (cof / mof / zif / prompt / amyloid)
    text: str              # full text appended to spec.tiers
    aliases: List[str]     # used by domain-relevance scoring


LAST_TIERS: Dict[str, LastTier] = {
    "cof": LastTier(
        key="cof",
        text="Covalent Organic Frameworks (COFs)",
        aliases=["COF", "COFs", "covalent organic framework",
                 "covalent-organic framework", "covalent organic frameworks"],
    ),
    "mof": LastTier(
        key="mof",
        text="Metal-Organic Frameworks (MOFs)",
        aliases=["MOF", "MOFs", "metal-organic framework",
                 "metal organic framework", "metal-organic frameworks"],
    ),
    "zif": LastTier(
        key="zif",
        text="Zeolitic Imidazolate Frameworks (ZIFs)",
        aliases=["ZIF", "ZIFs",
                 "zeolitic imidazolate framework",
                 "zeolitic imidazolate frameworks"],
    ),
    "prompt": LastTier(
        key="prompt",
        text="Prompt engineering for large language models",
        aliases=["prompt engineering", "prompting",
                 "LLM prompt", "LLM prompting",
                 "large language model", "large language models"],
    ),
    "amyloid": LastTier(
        key="amyloid",
        text="Amyloid production and APP processing",
        aliases=["amyloid", "amyloid-beta", "amyloid beta", "Abeta", "A-beta",
                 "APP", "APP processing", "amyloid precursor protein"],
    ),
}

_CHEM_DEFAULT_SOURCES = ("arxiv", "openalex", "nature", "core")
_AD_DEFAULT_SOURCES   = ("pubmed", "biorxiv", "openalex", "core")
_CS_DEFAULT_SOURCES   = ("arxiv", "openalex")

# ================== Domain registry ==================
DOMAINS: Dict[str, DomainSpec] = {
    "chem": DomainSpec(
        arxiv_cats=_CHEM_ARXIV_CATS,
        base_tiers=_CHEM_BASE_TIERS,
        default_sources=_CHEM_DEFAULT_SOURCES,
        valid_last_tiers=["cof", "mof", "zif"],
        default_last_tier=None,                       # MUST pass --last_tier
    ),
    "cs": DomainSpec(
        arxiv_cats=_CS_ARXIV_CATS,
        base_tiers=_CS_BASE_TIERS,
        default_sources=_CS_DEFAULT_SOURCES,
        valid_last_tiers=["prompt"],
        default_last_tier="prompt",
    ),
    "ad": DomainSpec(
        arxiv_cats=_AD_ARXIV_CATS,
        base_tiers=_AD_BASE_TIERS,
        default_sources=_AD_DEFAULT_SOURCES,
        valid_last_tiers=["amyloid"],
        default_last_tier="amyloid",
    ),
}

# =================== Helpers ===================
def get_spec(domain_key: str) -> DomainSpec:
    key = (domain_key or "").lower()
    if key not in DOMAINS:
        raise KeyError(
            f"Unknown domain '{domain_key}'. Available: {sorted(DOMAINS.keys())}"
        )
    return DOMAINS[key]

def parse_sources(value: str) -> Tuple[str, ...]:
    if not value:
        return ()
    out: List[str] = []
    for s in value.split(","):
        s = s.strip().lower()
        if not s:
            continue
        if s not in SEARCH_SOURCES:
            print(f"[domain_args] Unknown source '{s}' — skipping. "
                  f"Supported: {sorted(SEARCH_SOURCES.keys())}")
            continue
        if s not in out:
            out.append(s)
    return tuple(out)

# Legacy entry point retained for back-compat
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS.keys(),
                        default=(os.getenv("DOMAIN") or "cof").lower())
    parser.add_argument("--run-id", default=os.getenv("RUN_ID") or "")
    parser.add_argument("--sources", nargs="+",
                        choices=SEARCH_SOURCES.keys(),
                        default=None,
                        help="Space-separated list of search sources to use")
    parser.add_argument("--results-per-query", type=int, default=8)
    parser.add_argument("--target-per-round", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    return parser.parse_args()
