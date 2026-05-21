from pathlib import Path
from typing import List

def extract_pdf_text(pdf_path: Path) -> str:
    """Return the full text of a PDF as one string. Empty string on failure.

    Tries PyMuPDF (fitz) first because it matches Phase 1's convention and
    handles columns/figures better; falls back to pypdf so the script still
    runs in environments where fitz is unavailable.
    """
    text = _extract_with_fitz(pdf_path)
    if text.strip():
        return text
    return _extract_with_pypdf(pdf_path)

def _extract_with_fitz(pdf_path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        parts: List[str] = []
        for page in doc:
            try:
                parts.append(page.get_text() or "")
            except Exception:
                continue
        doc.close()
        return "\n".join(parts)
    except Exception as e:
        print(f"   ⚠️  fitz failed on {pdf_path.name}: {e}")
        return ""

def _extract_with_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception as e:
        print(f"   ⚠️  pypdf failed on {pdf_path.name}: {e}")
        return ""