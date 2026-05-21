import re
import requests
from typing import List
from requests.adapters import HTTPAdapter, Retry

# Keep a small set of stopwords for OpenAlex query cleaning
_OPENALEX_STOP = set("""
    a an the and or not of for to in on with without from by as at into over under
    vs versus about using via toward towards against between among through during
    is are was were be been being this that these those it its their his her our
    we you they i me my your yours our ours their theirs which who whom whose
    source
""".split())

# Clean query string for OpenAlex search
def clean_for_openalex(qstr: str) -> str:
    s = (qstr or "")
    # Strip arXiv-style fields and boolean glue
    s = re.sub(r"\b(?:ti|abs|cat):\S+", " ", s, flags=re.I)
    s = re.sub(r"\b(AND|OR|NOT)\b", " ", s, flags=re.I)
    # Remove quotes | Normalize dash variants | Collapse whitespace
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r'["“”\'()]+', " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    words = [
        w.lower()
        for w in re.split(r"[^a-zA-Z0-9_\-\+\.]+", s)
        if len(w) > 2
    ]
    words = [w for w in words if w not in _OPENALEX_STOP]

    out_terms: List[str] = []
    seen = set()
    for w in words:
        if w and w not in seen:
            out_terms.append(w)
            seen.add(w)
        if len(out_terms) >= 12:
            break

    q = " ".join(out_terms)
    return q[:200].strip()

# Pick a downloadable PDF link from an OpenAlex work
def _pick_openalex_pdf_url(work: dict) -> str | None:
    bol = work.get("best_oa_location") or {}
    pol = work.get("primary_location") or {}
    locs = [bol, pol] + (work.get("locations") or [])

    for loc in locs:
        u = (loc or {}).get("pdf_url")
        if u:
            return u

    for loc in locs:
        u = (loc or {}).get("url") or (loc or {}).get("landing_page_url")
        if u and u.lower().endswith(".pdf"):
            return u

    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    for loc in locs:
        u = (loc or {}).get("url") or (loc or {}).get("landing_page_url") or ""
        if "onlinelibrary.wiley.com" in u and doi:
            return f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}"
        if "royalsocietypublishing.org" in u and doi:
            return f"https://royalsocietypublishing.org/doi/pdf/{doi}"
        if "www.pnas.org" in u and doi:
            return f"https://www.pnas.org/doi/pdf/{doi}"
        if "science.org" in u and doi:
            return f"https://www.science.org/doi/pdf/{doi}"

    oa = work.get("open_access") or {}
    return oa.get("oa_url") or (pol.get("landing_page_url") if pol else None)

# Create a requests session for OpenAlex with retries
def openalex_session(mailto: str | None = None):
    s = requests.Session()
    ua = "paper-pipeline/0.1"
    if mailto:
        ua += f" ({mailto})"
    s.headers.update({"User-Agent": ua})
    retry = Retry(total=4, backoff_factor=0.8,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s
