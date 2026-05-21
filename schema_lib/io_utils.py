from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))