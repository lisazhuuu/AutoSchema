import re, os, time, requests, threading, xml.etree.ElementTree as ET
from typing import Dict, List
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime, timedelta

from expand_lib.utils import dedupe
from expand_lib.config import DEBUG
from expand_lib.openalex_search_helpers import (
    clean_for_openalex,
    _pick_openalex_pdf_url,
    openalex_session,
)

# ========== UA constants ==========
_API_UA = f"AutoSchema/0.1 ({os.getenv('NCBI_EMAIL', 'no-email')})"
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Chrome-fingerprint headers help PDF hosts behind Cloudflare (notably bioRxiv).
_CHROME_FETCH_HEADERS = {
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Linux"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# ========== NCBI session + polite GET ==========
PUBMED_SESSION = requests.Session()
PUBMED_SESSION.headers.update({
    "User-Agent": _API_UA,
    "Accept-Language": "en-US,en;q=0.9",
})

_NCBI_LAST_CALL = 0.0

def _ncbi_get(url: str, params: dict | None = None, timeout: int = 45):
    """Polite GET to NCBI eutils with rate limiting and retry on 429."""
    global _NCBI_LAST_CALL
    params = dict(params or {})
    email = (os.getenv("NCBI_EMAIL")
             or os.getenv("OPENALEX_MAILTO")
             or os.getenv("CROSSREF_MAILTO"))
    api_key = os.getenv("NCBI_API_KEY")
    if email:
        params.setdefault("email", email)
    params.setdefault("tool", "autoschema")
    if api_key:
        params.setdefault("api_key", api_key)

    min_gap = 0.20 if api_key else 0.55
    wait = min_gap - (time.time() - _NCBI_LAST_CALL)
    if wait > 0:
        time.sleep(wait)

    for attempt in range(3):
        _NCBI_LAST_CALL = time.time()
        try:
            r = PUBMED_SESSION.get(url, params=params, timeout=timeout,
                                   allow_redirects=True)
            if r.status_code == 429:
                time.sleep(min(30, 5 * (attempt + 1)))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == 2:
                print(f"[NCBI request error] {e}")
                return None
            time.sleep(3 * (attempt + 1))
    return None

# ========== Generic query cleaners ==========
_BOOL_RE  = re.compile(r"\b(AND|OR|NOT)\b", re.I)
_FIELD_RE = re.compile(r"\b(cat|ti|abs)\s*:\s*", re.I)

_STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its", "by",
    "on", "with", "without", "for", "at", "into", "over", "under", "via", "using",
    "paper", "papers", "article", "articles", "research", "study", "studies",
    "investigate", "investigates", "investigating", "examine", "examines", "examining",
    "search", "find", "retrieve", "focus", "focused", "including",
}

def _strip_query(qstr: str) -> str:
    if not qstr:
        return ""
    q = qstr.replace("’", "'").replace("‘", "'")
    q = _FIELD_RE.sub(" ", q)
    q = _BOOL_RE.sub(" ", q)
    q = re.sub(r"[^\w\s\-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _keywords(qstr: str, max_terms: int = 8) -> list[str]:
    q = _strip_query(qstr).lower()
    seen, out = set(), []
    for tok in q.split():
        if len(tok) <= 2 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_terms:
            break
    return out

def filter_query_for_pubmed(qstr: str, max_terms: int = 6) -> str:
    kws = _keywords(qstr, max_terms=max_terms)
    return " AND ".join(kws) if kws else ""

def filter_query_for_biorxiv(qstr: str, max_terms: int = 8) -> str:
    return " ".join(_keywords(qstr, max_terms=max_terms))

def clean_for_crossref(qstr: str) -> str:
    return _strip_query(qstr)

# ========== Domain relevance scoring (corpus-driven, no hardcoded keywords) ==========
_TIER_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

def _domain_strong_keywords(domain_spec) -> list:
    """Domain-relevance keywords derived from the active DomainSpec.tiers[-2:]"""
    if not domain_spec or not getattr(domain_spec, "tiers", None):
        return []
    parts = [t for t in domain_spec.tiers[-2:] if t]
    out: list = []
    for p in parts:
        for tok in _TIER_TOKEN_RE.findall(p.lower()):
            if len(tok) >= 4 and tok not in out:
                out.append(tok)
    return out

def score_domain_relevance(title: str, abstract: str, domain_spec) -> int:
    """Generic domain-relevance scorer; returns 0 if no spec / no tokens."""
    keywords = _domain_strong_keywords(domain_spec)
    if not keywords:
        return 0
    title_l = (title or "").lower()
    text = f"{title_l} {(abstract or '').lower()}"
    score = 0
    for k in keywords:
        if k in title_l:
            score += 6
        elif k in text:
            score += 3
    return score

# ==================== Search: arXiv ====================
_ARXIV_CLIENT = None
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL = 0.0
_ARXIV_MIN_GAP   = float(os.getenv("ARXIV_MIN_GAP", "5.0"))   # >=3s；5s 更安全
_ARXIV_MAX_PAGE  = int(os.getenv("ARXIV_MAX_PAGE", "25"))     # 小分页，别要 100
_ARXIV_FAIL_RETRIES = int(os.getenv("ARXIV_FAIL_RETRIES", "2"))  # 快速失败，不死磕
_ARXIV_429_TRIP  = int(os.getenv("ARXIV_429_TRIP", "3"))     # 连续 N 次失败就熔断
_ARXIV_DISABLED  = os.getenv("ARXIV_DISABLE", "").lower() in ("1", "true", "yes")
_ARXIV_FAIL_STREAK = 0

def _get_arxiv_client():
    global _ARXIV_CLIENT
    if _ARXIV_CLIENT is None:
        import arxiv
        _ARXIV_CLIENT = arxiv.Client(page_size=_ARXIV_MAX_PAGE,
                                     delay_seconds=_ARXIV_MIN_GAP,
                                     num_retries=2)
    return _ARXIV_CLIENT


def _arxiv_polite_wait():
    global _ARXIV_LAST_CALL
    with _ARXIV_LOCK:
        gap = _ARXIV_MIN_GAP - (time.time() - _ARXIV_LAST_CALL)
        if gap > 0:
            time.sleep(gap)
        _ARXIV_LAST_CALL = time.time()
        
def search_arxiv(qstr: str, max_results: int, arxiv_cats: List[str],
                 domain_spec=None) -> List[Dict]:
    global _ARXIV_DISABLED, _ARXIV_FAIL_STREAK
    import arxiv

    # Circuit breaker: once arXiv has rate-limited us repeatedly, stop hitting
    # it for the rest of the process and let the other sources carry the round.
    if _ARXIV_DISABLED:
        return []

    cat_block = " OR ".join(f"cat:{c}" for c in (arxiv_cats or []))
    if cat_block and "cat:" not in (qstr or "").lower():
        queries = [f"({qstr}) AND ({cat_block})", qstr]
    else:
        queries = [qstr]

    client = _get_arxiv_client()
    page_size = max(1, min(max_results, _ARXIV_MAX_PAGE))   # cap at ~25, not 100

    hits: List[Dict] = []
    for q in queries:
        if DEBUG:
            print(f"[arxiv] {q}")
        got = False
        for attempt in range(_ARXIV_FAIL_RETRIES + 1):      # e.g. 3 tries total
            try:
                _arxiv_polite_wait()
                search = arxiv.Search(query=q, max_results=page_size,
                                      sort_by=arxiv.SortCriterion.Relevance)
                batch: List[Dict] = []
                for r in client.results(search):
                    title = r.title or ""
                    abstract = r.summary or ""
                    batch.append({
                        "source": "arxiv",
                        "id": r.get_short_id(),
                        "title": title,
                        "authors": [a.name for a in r.authors],
                        "year": r.published.year if r.published else None,
                        "doi": r.doi,
                        "abstract": abstract,
                        "url": r.entry_id,
                        "pdf_url": getattr(r, "pdf_url", None),
                        "domain_score": score_domain_relevance(title, abstract, domain_spec),
                    })
                hits.extend(batch)
                _ARXIV_FAIL_STREAK = 0                       # success resets streak
                got = True
                break
            except Exception as e:
                msg = str(e)
                rate_limited = ("429" in msg) or ("503" in msg)
                if rate_limited:
                    _ARXIV_FAIL_STREAK += 1
                # Trip the breaker on a sustained streak (across all calls).
                if _ARXIV_FAIL_STREAK >= _ARXIV_429_TRIP:
                    _ARXIV_DISABLED = True
                    print(f"[arxiv] rate-limited {_ARXIV_FAIL_STREAK}x in a row "
                          f"-> DISABLING arxiv for the rest of this run; "
                          f"other sources will carry the round. "
                          f"(retry arxiv later or set --sources without arxiv)")
                    return hits           # bail out immediately, no more waiting
                wait = 15 * (attempt + 1)  # short: 15,30 (not 60..300)
                print(f"[arxiv error] {e} -> retry in {wait}s "
                      f"(attempt {attempt + 1}/{_ARXIV_FAIL_RETRIES + 1})")
                time.sleep(wait)
        if got and hits:
            break

    return dedupe(hits)[:max_results]

# ==================== Search: OpenAlex ====================
def search_openalex(qstr: str, max_results: int, mailto: str | None = None,
                    domain_spec=None) -> List[Dict]:
    mailto = (mailto
              or os.getenv("OPENALEX_MAILTO")
              or os.getenv("CROSSREF_MAILTO")
              or "noreply@example.com")
    sess = openalex_session(mailto)
    base_q = clean_for_openalex(qstr)
    base_url = "https://api.openalex.org/works"

    filter_variants = [
        "type:journal-article,open_access.is_oa:true",
        "open_access.is_oa:true",
        "",
    ]

    out: List[Dict] = []
    for filt in filter_variants:
        params = {
            "search": base_q,
            "per_page": max(max_results, 25),
            "page": 1,
            "mailto": mailto,
        }
        if filt:
            params["filter"] = filt
        try:
            r = sess.get(base_url, params=params, timeout=25)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[OpenAlex] {e}; filter={filt!r}")
            continue

        for w in (r.json() or {}).get("results", []):
            doi = (w.get("doi") or "").replace("https://doi.org/", "") if w.get("doi") else None
            title = w.get("title") or ""
            out.append({
                "source": "openalex",
                "id": (w.get("ids") or {}).get("openalex") or w.get("id"),
                "title": title,
                "authors": [a["author"]["display_name"]
                            for a in (w.get("authorships") or [])
                            if a.get("author")],
                "year": w.get("publication_year"),
                "doi": doi,
                "abstract": None,
                "url": w.get("id"),
                "pdf_url": _pick_openalex_pdf_url(w),
                "domain_score": score_domain_relevance(title, "", domain_spec),
            })

        if len(out) >= max_results:
            break

    return dedupe(out)[:max_results]

# ==================== Search Nature ====================
def _search_crossref_member(qstr: str, max_results: int, member: str,
                            mailto: str | None, domain_spec=None) -> List[Dict]:
    base = clean_for_crossref(qstr)
    headers = {"User-Agent":
               f"AutoSchema/0.1 ({'mailto:' + mailto if mailto else 'no-mailto'})"}
    params = {
        "query": base,
        "filter": f"member:{member},type:journal-article",
        "rows": max(max_results, 25),
    }
    if mailto:
        params["mailto"] = mailto

    try:
        r = requests.get("https://api.crossref.org/works",
                         params=params, timeout=30, headers=headers)
        r.raise_for_status()
    except Exception as e:
        print(f"[Crossref member={member} error] {e}")
        return []

    out: List[Dict] = []
    for it in (r.json() or {}).get("message", {}).get("items", []) or []:
        doi = it.get("DOI")
        url = it.get("URL") or (f"https://doi.org/{doi}" if doi else None)
        title = (it.get("title") or [None])[0]
        authors = [
            (" ".join(filter(None, [a.get("given", ""), a.get("family", "")])).strip()
             or a.get("name"))
            for a in (it.get("author") or []) if a
        ]
        dp = (((it.get("issued") or {}).get("date-parts") or [[]])[0] or [])
        year = dp[0] if dp else None
        pdf_url = None
        for lk in (it.get("link") or []):
            ct = (lk.get("content-type") or lk.get("content_type") or "").lower()
            if "pdf" in ct:
                pdf_url = lk.get("URL") or lk.get("url")
                break
        out.append({
            "source": "crossref", "id": doi or url, "title": title or "",
            "authors": authors, "year": year, "doi": doi,
            "abstract": None, "url": url, "pdf_url": pdf_url,
            "domain_score": score_domain_relevance(title or "", "", domain_spec),
        })
    return dedupe(out)[:max_results]

def search_crossref_nature(qstr: str, max_results: int, mailto: str | None = None,
                           domain_spec=None) -> List[Dict]:
    return _search_crossref_member(qstr, max_results, "297", mailto,
                                   domain_spec=domain_spec)

# ==================== Search: bioRxiv ====================
def search_biorxiv(qstr: str, max_results: int, domain_spec=None,
                   scan_days: int = 730, max_pages: int = 30) -> List[Dict]:
    keywords = filter_query_for_biorxiv(qstr).split()
    if not keywords:
        return []

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=scan_days)).strftime("%Y-%m-%d")
    print(f"[bioRxiv] keywords={keywords[:8]} window={start_date}..{end_date}")

    candidates: List[Dict] = []
    cursor = 0
    for _ in range(max_pages):
        url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/{cursor}/json"
        try:
            r = requests.get(url, timeout=60)
            data = (r.json().get("collection", []) or [])
        except Exception as e:
            print(f"[bioRxiv error] {e}")
            break
        if not data:
            break

        for item in data:
            title = item.get("title") or ""
            abstract = item.get("abstract") or ""
            d_score = score_domain_relevance(title, abstract, domain_spec)
            if d_score <= 0:
                continue

            content = (title + " " + abstract).lower()
            kw_score = sum(content.count(k) for k in keywords)
            doi = item.get("doi")
            candidates.append({
                "_dom": d_score,
                "_kw": kw_score,
                "source": "biorxiv",
                "id": doi,
                "title": title,
                "authors": [a.strip() for a in (item.get("authors") or "").split(";") if a.strip()],
                "year": int(item.get("date", "")[:4]) if item.get("date") else None,
                "doi": doi,
                "abstract": abstract or None,
                "url": f"https://doi.org/{doi}" if doi else None,
                "pdf_url": f"https://www.biorxiv.org/content/{doi}.full.pdf" if doi else None,
                "domain_score": d_score,
            })

        if len(candidates) >= max_results * 5:
            break
        cursor += 100
        time.sleep(0.3)

    candidates.sort(key=lambda x: (x.get("_dom", 0), x.get("_kw", 0)), reverse=True)
    for c in candidates:
        c.pop("_dom", None)
        c.pop("_kw", None)

    print(f"[bioRxiv] domain-relevant: {len(candidates)} (returning top {max_results})")
    return candidates[:max_results]

def download_biorxiv_hit(hit: dict, download_dir, filename: str) -> bool:
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    dest = download_dir / filename

    doi = hit.get("doi") or ""
    if not doi:
        print("  --> [Fail biorxiv] no DOI on hit")
        return False

    referer = f"https://www.biorxiv.org/content/{doi}"
    direct_urls = [
        f"https://www.biorxiv.org/content/{doi}.full.pdf",
        f"https://www.biorxiv.org/content/{doi}v1.full.pdf",
        f"https://www.biorxiv.org/content/{doi}v2.full.pdf",
    ]
    for url in direct_urls:
        if _download_pdf_url(url, dest, referer=referer):
            print(f"  --> [Success] {dest.name}")
            return True

    print(f"  --> [Fail biorxiv] {doi}")
    return False

# ==================== Search: CORE ====================
def search_core(qstr: str, max_results: int, api_key: str | None,
                domain_spec=None):
    if not api_key:
        # Caller (expand_round.py) is expected to skip CORE entirely if no key
        # is configured, but be defensive in case the function is called directly.
        print("[CORE] Missing API key — skipping. Set CORE_API_KEY.")
        return []
    q = _strip_query(qstr)[:250]
    if not q:
        return []

    try:
        r = requests.get(
            "https://api.core.ac.uk/v3/search/works",
            headers={"Authorization": f"Bearer {api_key}",
                     "Accept": "application/json"},
            params={"q": q, "limit": min(max_results * 3, 100), "offset": 0},
            timeout=30,
        )
        if r.status_code in (401, 403):
            print(f"[CORE] auth error {r.status_code} — check CORE_API_KEY")
            return []
        if r.status_code == 429:
            return None
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"[CORE error] {e}")
        return []

    hits: List[Dict] = []
    for item in payload.get("results", []) or []:
        title = item.get("title")
        abstract = item.get("abstract") or item.get("description") or ""
        authors_raw = item.get("authors") or []
        authors: List[str] = []
        for a in authors_raw:
            if isinstance(a, dict):
                if a.get("name"):
                    authors.append(a["name"])
            elif isinstance(a, str):
                authors.append(a)

        landing = item.get("downloadUrl") or item.get("fullTextLink") or item.get("url")
        pdf_candidates = [item.get("downloadUrl"),
                          item.get("pdfUrl"),
                          item.get("fullTextLink")]
        sfu = item.get("sourceFulltextUrls")
        if isinstance(sfu, list):
            for u in sfu:
                if isinstance(u, str):
                    pdf_candidates.append(u)
        pdf_url = next((u for u in pdf_candidates
                        if isinstance(u, str) and u), None)

        hits.append({
            "source": "core",
            "id": item.get("id") or item.get("coreId"),
            "title": title,
            "authors": authors,
            "year": item.get("year") or item.get("publishedYear"),
            "doi": item.get("doi"),
            "abstract": abstract or None,
            "url": landing,
            "pdf_url": pdf_url,
            "domain_score": score_domain_relevance(title or "", abstract, domain_spec),
        })

    hits.sort(key=lambda x: (x.get("domain_score", 0),
                             1 if x.get("pdf_url") else 0),
              reverse=True)
    return hits[:max_results]

# ==================== Search: PubMed ====================
def _extract_pmcid_from_esummary_doc(doc: dict) -> str | None:
    for item in doc.get("articleids", []) or []:
        if (item.get("idtype") or "").lower() == "pmc":
            v = item.get("value") or item.get("id")
            if v:
                v = str(v)
                return v if v.startswith("PMC") else f"PMC{v}"
    return None

def _idconv_pmids_to_pmcids(pmids: list[str], mailto: str | None = None) -> dict:
    if not pmids:
        return {}
    params = {
        "format": "json",
        "ids": ",".join(str(x) for x in pmids),
        "tool": "autoschema",
    }
    email = mailto or os.getenv("NCBI_EMAIL") or os.getenv("OPENALEX_MAILTO")
    if email:
        params["email"] = email
    r = _ncbi_get(
        "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
        params=params, timeout=45,
    )
    if not r:
        return {}
    out: Dict[str, str] = {}
    try:
        for rec in r.json().get("records", []):
            pmid = str(rec.get("pmid") or "")
            pmcid = rec.get("pmcid")
            if pmid and pmcid:
                out[pmid] = pmcid
    except Exception as e:
        print(f"[PMC ID Converter parse error] {e}")
    return out

def search_pubmed(qstr: str, max_results: int, mailto: str | None = None,
                  domain_spec=None) -> List[Dict]:
    if not os.getenv("NCBI_EMAIL"):
        print("[PubMed] NCBI_EMAIL not set — proceeding without identity "
              "header (NCBI may rate-limit aggressively).")

    query = filter_query_for_pubmed(qstr, max_terms=6)
    if not query:
        return []
    print(f"[PubMed Search] {query}")

    r = _ncbi_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": f"({query}) AND \"pubmed pmc\"[sb]",
            "retmax": max(max_results * 3, 30),
            "retmode": "json",
            "sort": "relevance",
        },
        timeout=45,
    )
    if not r:
        return []

    try:
        pmids = r.json().get("esearchresult", {}).get("idlist", []) or []
    except Exception as e:
        print(f"[PubMed esearch parse error] {e}")
        return []

    if not pmids:
        print("[PubMed] No IDs found.")
        return []

    sr = _ncbi_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        timeout=45,
    )
    summary = (sr.json().get("result", {}) if sr else {}) or {}
    pmcid_map = _idconv_pmids_to_pmcids(pmids, mailto=mailto)

    hits: List[Dict] = []
    for pmid in pmids:
        doc = summary.get(str(pmid), {}) or {}
        pmcid = _extract_pmcid_from_esummary_doc(doc) or pmcid_map.get(str(pmid))
        if not pmcid:
            continue

        article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

        year = None
        m = re.match(r"(\d{4})", doc.get("pubdate") or doc.get("epubdate") or "")
        if m:
            try:
                year = int(m.group(1))
            except Exception:
                pass

        authors = [a.get("name") for a in (doc.get("authors") or []) if a.get("name")]
        title = doc.get("title") or ""

        hits.append({
            "source": "pubmed",
            "id": pmid,
            "pmid": pmid,
            "pmcid": pmcid,
            "title": title,
            "authors": authors,
            "year": year,
            "doi": None,
            "abstract": None,
            "url": article_url,
            "pdf_url": None,
            "article_url": article_url,
            "domain_score": score_domain_relevance(title, "", domain_spec),
        })
        if len(hits) >= max_results:
            break
    return hits

_META_PDF_RE  = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
    re.I,
)
_META_PDF_RE2 = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
    re.I,
)
_PMC_PDF_LINK_RE = re.compile(
    r'(/articles/PMC\d+/pdf/[^"\'\s>]+\.pdf)', re.I,
)

def _find_pmc_pdf_via_scrape(article_url: str) -> str | None:
    if not article_url:
        return None
    try:
        r = PUBMED_SESSION.get(
            article_url,
            headers={
                "User-Agent": _BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=45, allow_redirects=True,
        )
    except Exception as e:
        print(f"  [scrape error] {e}")
        return None
    if not r.ok:
        print(f"  [scrape] article page status={r.status_code}")
        return None

    html = r.text or ""
    m = _META_PDF_RE.search(html) or _META_PDF_RE2.search(html)
    if m:
        return m.group(1)
    m = _PMC_PDF_LINK_RE.search(html)
    if m:
        return urljoin(article_url, m.group(1))
    return None

def _get_pmc_pdf_candidates(article_url: str) -> list[str]:
    """Ranked PDF URL candidates for a PMC article."""
    m = re.search(r"/articles/(PMC\d+)/?", article_url or "")
    if not m:
        return []
    pmcid = m.group(1)

    out: list[str] = []
    # 1) Europe PMC -- primary
    out.append(f"https://europepmc.org/articles/{pmcid}?pdf=render")
    # 2) Scrape PMC article page for the real PDF link
    scraped = _find_pmc_pdf_via_scrape(article_url)
    if scraped and scraped not in out:
        out.append(scraped)
    # 3) OA API alternates
    r = _ncbi_get(
        "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
        params={"id": pmcid}, timeout=45,
    )
    if r:
        try:
            root = ET.fromstring(r.text)
            for link in root.findall(".//link"):
                fmt = (link.attrib.get("format") or "").lower()
                href = (link.attrib.get("href") or "").strip()
                if not href:
                    continue
                if href.lower().startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
                    href = "https://ftp.ncbi.nlm.nih.gov/" + href[len("ftp://ftp.ncbi.nlm.nih.gov/"):]
                h = href.lower()
                if fmt != "pdf" and ".pdf" not in h:
                    continue
                if any(x in h for x in ["supp", "supplement", "s001", "s002",
                                        "mmc", "moesm", "appendix"]):
                    continue
                if href not in out:
                    out.append(href)
        except Exception as e:
            print(f"  [OA API parse error] {e}")
    return out

def _download_pdf_url(pdf_url: str, dest: Path, referer: str | None = None,
                      return_status: bool = False):
    def _ret(success: bool, status: int):
        return (success, status) if return_status else success

    if not pdf_url:
        return _ret(False, -1)
    if pdf_url.lower().startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        pdf_url = "https://ftp.ncbi.nlm.nih.gov/" + pdf_url[len("ftp://ftp.ncbi.nlm.nih.gov/"):]

    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                  "application/pdf,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        **_CHROME_FETCH_HEADERS,
    }
    if referer:
        headers["Referer"] = referer

    try:
        r = PUBMED_SESSION.get(pdf_url, headers=headers, timeout=90,
                               allow_redirects=True)
    except Exception as e:
        print(f"  [Download Error] {e}")
        return _ret(False, -1)

    body = r.content or b""
    ctype = (r.headers.get("content-type") or "").lower()
    if r.ok and (body.startswith(b"%PDF") or ("pdf" in ctype and len(body) > 1000)):
        dest.write_bytes(body)
        return _ret(True, r.status_code)

    print(f"  [Download failed] status={r.status_code}, ctype={ctype}, url={pdf_url}")
    return _ret(False, r.status_code)

def download_pubmed_or_pmc_hit(hit: dict, download_dir, filename: str) -> bool:
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    dest = download_dir / filename

    article_url = hit.get("article_url") or hit.get("url")
    candidates = _get_pmc_pdf_candidates(article_url)
    if hit.get("pdf_url") and hit["pdf_url"] not in candidates:
        candidates.insert(0, hit["pdf_url"])

    if not candidates:
        print(f"  --> [Fail] no candidates for {hit.get('pmcid') or hit.get('pmid')}")
        return False

    for url in candidates:
        if _download_pdf_url(url, dest, referer=article_url):
            print(f"  --> [Success] {dest.name}")
            return True

    print(f"  --> [Fail] {hit.get('pmcid') or hit.get('pmid')}")
    return False
