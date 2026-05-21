from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optionally load .env so this module also works when imported standalone.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ======================== LLMConfig ========================
@dataclass
class LLMConfig:
    provider: str = "azure_openai"           # azure_openai | openai

    # ----- Chat -----
    chat_model: str = "gpt-5.4"               # used when provider == "openai"
    chat_deployment: str = "gpt-5.4"          # used when provider == "azure_openai"
    chat_api_key: Optional[str] = None
    chat_endpoint: Optional[str] = None       # Azure only
    chat_version: str = "2024-12-01-preview"  # Azure only

    # ----- Embeddings -----
    embedding_model: str = "text-embedding-3-large"
    embedding_deployment: str = "text-embedding-3-large"
    embedding_api_key: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embedding_version: str = "2024-12-01-preview"

    # ----- Filesystem -----
    output_dir: str = "output"
    text_dir: str = "output/texts"

# ======================== Config loading ========================
_CONFIG_PATH_ENV = "AUTOSCHEMA_CONFIG"
_DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"

def _resolve_config_path() -> Path:
    explicit = os.getenv(_CONFIG_PATH_ENV)
    return Path(explicit) if explicit else _DEFAULT_CONFIG_PATH

def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}

def load_config() -> LLMConfig:
    """Materialize an LLMConfig from YAML defaults + environment overrides."""
    cfg = LLMConfig()

    yaml_data = _load_yaml_config(_resolve_config_path())
    for key, val in yaml_data.items():
        if hasattr(cfg, key) and val is not None:
            setattr(cfg, key, val)

    # Env overrides (highest precedence). Secrets always come from env / .env.
    cfg.provider = os.getenv("LLM_PROVIDER", cfg.provider).lower()

    # Chat
    cfg.chat_model      = os.getenv("CHAT_MODEL",      cfg.chat_model)
    cfg.chat_deployment = os.getenv("CHAT_DEPLOYMENT", cfg.chat_deployment)
    cfg.chat_api_key    = os.getenv("CHAT_API_KEY",    cfg.chat_api_key)
    cfg.chat_endpoint   = os.getenv("CHAT_ENDPOINT",   cfg.chat_endpoint)
    cfg.chat_version    = os.getenv("CHAT_VERSION",    cfg.chat_version)

    # Embeddings
    cfg.embedding_model      = os.getenv("EMBEDDING_MODEL",      cfg.embedding_model)
    cfg.embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT", cfg.embedding_deployment)
    cfg.embedding_api_key    = os.getenv("EMBEDDING_API_KEY",    cfg.embedding_api_key)
    cfg.embedding_endpoint   = os.getenv("EMBEDDING_ENDPOINT",   cfg.embedding_endpoint)
    cfg.embedding_version    = os.getenv("EMBEDDING_VERSION",    cfg.embedding_version)

    # Filesystem
    cfg.output_dir = os.getenv("AUTOSCHEMA_OUTPUT_DIR", cfg.output_dir)
    cfg.text_dir   = os.getenv("AUTOSCHEMA_TEXT_DIR",   cfg.text_dir)
    return cfg

# ======================== Lazy singletons ========================
_CONFIG: Optional[LLMConfig] = None
_CHAT_CLIENT = None
_EMBED_CLIENT = None

def get_config() -> LLMConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG

def _require_chat_credentials(cfg: LLMConfig) -> None:
    if not cfg.chat_api_key:
        raise RuntimeError(
            "Missing CHAT_API_KEY. Set it in your .env (see .env.example)."
        )
    if cfg.provider == "azure_openai" and not cfg.chat_endpoint:
        raise RuntimeError(
            "CHAT_ENDPOINT is required when provider=azure_openai. "
            "Set it in your .env."
        )

def _require_embedding_credentials(cfg: LLMConfig) -> None:
    api_key  = cfg.embedding_api_key  or cfg.chat_api_key
    endpoint = cfg.embedding_endpoint or cfg.chat_endpoint
    if not api_key:
        raise RuntimeError(
            "Missing EMBEDDING_API_KEY (and no CHAT_API_KEY fallback). "
            "Set it in your .env."
        )
    if cfg.provider == "azure_openai" and not endpoint:
        raise RuntimeError(
            "EMBEDDING_ENDPOINT (or CHAT_ENDPOINT fallback) is required "
            "when provider=azure_openai."
        )

def _build_chat_client(cfg: LLMConfig):
    if cfg.provider == "azure_openai":
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=cfg.chat_api_key,
            api_version=cfg.chat_version,
            azure_endpoint=cfg.chat_endpoint,
        )
    if cfg.provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=cfg.chat_api_key)
    raise RuntimeError(
        f"Unknown LLM provider '{cfg.provider}'. Supported: azure_openai, openai."
    )

def _build_embedding_client(cfg: LLMConfig):
    api_key  = cfg.embedding_api_key  or cfg.chat_api_key
    endpoint = cfg.embedding_endpoint or cfg.chat_endpoint
    version  = cfg.embedding_version  or cfg.chat_version
    if cfg.provider == "azure_openai":
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=api_key,
            api_version=version,
            azure_endpoint=endpoint,
        )
    if cfg.provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    raise RuntimeError(
        f"Unknown LLM provider '{cfg.provider}'. Supported: azure_openai, openai."
    )

def get_chat_client():
    global _CHAT_CLIENT
    cfg = get_config()
    _require_chat_credentials(cfg)
    if _CHAT_CLIENT is None:
        _CHAT_CLIENT = _build_chat_client(cfg)
    return _CHAT_CLIENT

def get_embedding_client():
    global _EMBED_CLIENT
    cfg = get_config()
    _require_embedding_credentials(cfg)
    if _EMBED_CLIENT is None:
        _EMBED_CLIENT = _build_embedding_client(cfg)
    return _EMBED_CLIENT

def _chat_model_arg(cfg: LLMConfig) -> str:
    return cfg.chat_deployment if cfg.provider == "azure_openai" else cfg.chat_model

def _embedding_model_arg(cfg: LLMConfig) -> str:
    return cfg.embedding_deployment if cfg.provider == "azure_openai" else cfg.embedding_model

# ======================== Public helpers ========================
def llm_chat(messages: List[Dict[str, str]], force_json: bool = False) -> str:
    """Provider-agnostic chat completion with retries."""
    from openai import APIError, RateLimitError, APIConnectionError
    try:                                                        # SDK >= 1.0
        from openai import APITimeoutError as _TimeoutErr
    except ImportError:                                         # legacy SDK
        from openai import Timeout as _TimeoutErr

    cfg = get_config()
    client = get_chat_client()

    kwargs: Dict[str, Any] = {
        "model": _chat_model_arg(cfg),
        "messages": messages,
    }
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except (APIError, RateLimitError, APIConnectionError, _TimeoutErr) as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 60 if isinstance(e, RateLimitError) else 2 ** (attempt + 2)
                print(
                    f"[LLM Retry] API call failed ({e}). "
                    f"Waiting {wait_time}s. (Attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait_time)
            else:
                print(f"[ERROR] LLM API call failed after {MAX_RETRIES} retries: {e}")
                return json.dumps({"queries": []})
        except Exception as e:
            print(f"[ERROR] LLM API call failed (unexpected): {e}")
            return json.dumps({"queries": []})

def embedding_model_name() -> str:
    """Resolved model / deployment name to pass to ``client.embeddings.create``."""
    return _embedding_model_arg(get_config())