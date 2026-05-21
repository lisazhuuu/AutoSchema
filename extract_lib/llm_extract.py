from __future__ import annotations

import json,os, re, sys, time
from pathlib import Path
from typing import Any, Dict, Optional

from extract_lib.config import (
    DEFAULT_MODEL_DEPLOYMENT,
    LLM_RETRIES,
    LLM_RETRY_BACKOFF_SEC,
    LLM_TEMPERATURE,
    MAX_COMPLETION_TOKENS,
)

# Locate the repo's prompt_lib.llm_client
def _load_repo_llm_client():
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "prompt_lib" / "llm_client.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            break
    try:
        from prompt_lib import llm_client  # type: ignore
        return llm_client
    except Exception:
        return None

# Client construction 
def build_chat_client():
    mod = _load_repo_llm_client()
    if mod is not None:
        # Lazy: get_chat_client() validates credentials and builds Azure/OpenAI.
        return mod.get_chat_client()

    # ---- Fallback: self-contained, same env names as prompt_lib ----
    provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()
    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHAT_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY (LLM_PROVIDER=openai). Set it in your "
                ".env. See the repo .env.example."
            )
        return OpenAI(api_key=api_key)

    # Default: azure_openai
    from openai import AzureOpenAI
    api_key = os.getenv("CHAT_API_KEY") or os.getenv("AZURE_OPENAI_CHAT_API_KEY")
    endpoint = os.getenv("CHAT_ENDPOINT") or os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
    api_version = (os.getenv("CHAT_VERSION")
                   or os.getenv("AZURE_OPENAI_CHAT_API_VERSION")
                   or "2024-12-01-preview")
    if not api_key or not endpoint:
        raise RuntimeError(
            "Missing CHAT_API_KEY / CHAT_ENDPOINT (LLM_PROVIDER=azure_openai). "
            "Set them in your .env. See the repo .env.example."
        )
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint,
                       api_version=api_version)

# Backward-compatible alias
def build_azure_client():
    return build_chat_client()

def get_model_deployment() -> str:
    mod = _load_repo_llm_client()
    if mod is not None:
        try:
            cfg = mod.get_config()
            name = (cfg.chat_deployment if cfg.provider == "azure_openai"
                    else cfg.chat_model)
            if name:
                return name
        except Exception:
            pass
    return (os.getenv("CHAT_DEPLOYMENT")
            or os.getenv("CHAT_MODEL")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or DEFAULT_MODEL_DEPLOYMENT)

# JSON parsing
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

def _robust_json_parse(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = _CODE_FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    candidate = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def _sanitize(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "replace").decode("utf-8")

# Main entry
def call_extract(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    paper_id: str,
) -> Optional[Dict[str, Any]]:
    """Run a single extraction call. Returns parsed JSON or None on failure."""
    messages = [
        {"role": "system", "content": _sanitize(system_prompt)},
        {"role": "user", "content": _sanitize(user_prompt)},
    ]

    last_err: Optional[str] = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _robust_json_parse(raw)
            if parsed is not None:
                # stamp the paper_id so downstream code can rely on it
                parsed["paper_id"] = paper_id
                return parsed
            last_err = "json_parse_failed"
            print(f"   ⚠️  attempt {attempt}: JSON parse failed for {paper_id}")
        except Exception as e:
            last_err = repr(e)
            print(f"   ⚠️  attempt {attempt} failed for {paper_id}: {e}")
        time.sleep(LLM_RETRY_BACKOFF_SEC * attempt)

    print(f"   ❌ giving up on {paper_id} after {LLM_RETRIES} attempts (last: {last_err})")
    return None