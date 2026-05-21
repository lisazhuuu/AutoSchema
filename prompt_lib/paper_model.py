from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
# Other helper
from prompt_lib.domain import OUT_DIR

# Convert pdf to txt file and return the txt file path
def pdf_to_text(pdf_path: Path) -> str:
    import fitz 
    
    text_dir = OUT_DIR / "texts"; text_dir.mkdir(exist_ok=True)
    doc = fitz.open(pdf_path)
    chunks = [p.get_text() for p in doc]
    # Save to text file
    outp = text_dir / (pdf_path.stem + ".txt")
    outp.write_text("\n".join(chunks), encoding="utf-8")
    
    return str(outp)

# The paper data structure and its metadata fields
@dataclass
class Paper: 
    paper_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    venue: Optional[str] = None # Where it was published
    doi: Optional[str] = None # Digital Object Identifier (if available)
    source: str = "seed"
    raw_text_path: Optional[str] = None
    extracted: Dict[str, Any] = None # A dictionary of structured data extracted from the paper