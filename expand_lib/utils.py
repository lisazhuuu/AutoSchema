import re, os
from pathlib import Path
from typing import List, Dict, Set
from urllib.parse import urlparse
from datetime import datetime

from expand_lib.preprocessing import norm_title

# Remove duplicate records based on DOI, ID, or normalized title
def dedupe(records: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    out: List[Dict] = []
    for r in records:
        key = r.get("doi") or r.get("id") or norm_title(r.get("title", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out

# Site tag used as a filename suffix
def site_tag(rec: dict, collapse_springer_nature: bool = True) -> str:
    u    = rec.get("pdf_url") or rec.get("url") or ""
    host = (urlparse(u).hostname or "").lower()
    doi  = (rec.get("doi") or "").lower()

    if collapse_springer_nature:
        if doi.startswith(("10.1038/", "10.1007/", "10.1186/")):
            return "nature"
        if any(h in host for h in [
            "nature.com", "springernature.com",
            "link.springer.com", "biomedcentral.com", "bmc.com",
        ]):
            return "nature"

    if rec.get("source") == "openalex":
        return "openalex"
    if "arxiv.org" in host:
        return "arxiv"
    if "biorxiv.org" in host or doi.startswith("10.1101/"):
        return "biorxiv"
    if "ncbi.nlm.nih.gov" in host or "europepmc.org" in host:
        return "pubmed"
    if "core.ac.uk" in host:
        return "core"

    if host and host != "doi.org":
        parts = host.split(".")
        return parts[-2] if len(parts) >= 2 else host
    return (rec.get("source") or "unknown").lower()

def fname_with_tag(rec: dict) -> str:
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "_",
                  (rec.get("title") or rec.get("id") or "paper")).strip("_")
    tag = site_tag(rec)
    suffix = f"__{tag}.pdf"
    return (base[: max(1, 120 - len(suffix))] + suffix)

def download_pdf(url: str, dest: Path) -> bool:
    """Generic direct PDF download. Source-specific downloaders
    (PMC, bioRxiv, etc.) live in expand_lib.searches and are called
    explicitly by expand_round.py."""
    import requests

    headers = {
        "User-Agent": f"paper-pipeline/0.1 ({os.getenv('OPENALEX_MAILTO') or 'noreply@example.com'})",
    }
    try:
        with requests.get(
            url, stream=True, timeout=60, allow_redirects=True, headers=headers,
        ) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(1024 * 64):
                    f.write(chunk)
        return True
    except Exception:
        return False

def delete_empty_folders(path: Path):
    if not path.is_dir():
        return
    if not any(path.iterdir()):
        path.rmdir()
        print(f"Deleted empty folder: {path}")

def _is_valid_pdf(p: Path) -> bool:
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            head = f.read(5)
            if size > 2048:
                f.seek(size - 2048)
            tail = f.read()
        return head.startswith(b"%PDF-") and (b"%%EOF" in tail)
    except Exception:
        return False

def delete_invalid_pdfs(path):
    p = Path(path)
    deleted: List[str] = []
    try:
        if p.is_file():
            if not _is_valid_pdf(p):
                p.unlink()
                deleted.append(p.name)
        elif p.is_dir():
            for f in p.glob("*.pdf"):
                if not _is_valid_pdf(f):
                    f.unlink()
                    deleted.append(f.name)
    except Exception as e:
        print(f"[delete_invalid_pdfs] error: {e}")
    print(f"deleted {len(deleted)} bad pdfs")
    return deleted

def make_unique_run_dir(base: Path, prefix: str = "round") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
