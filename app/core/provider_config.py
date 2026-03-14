"""
Remote LLM / Reranker provider configuration.

Persisted to ``<runtime_root>/provider_config.json``.
Loaded once at startup and mutated at runtime via the ``/providers`` router.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class RemoteEndpointConfig(BaseModel):
    """Configuration for one remote endpoint (LLM or Reranker)."""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # Provider hint – purely informational, the HTTP layer always uses OpenAI-compat format
    provider_hint: str = "openai"


class ProviderSettings(BaseModel):
    """Top-level provider configuration."""
    # "local" = use local llama-server; "remote" = use cloud endpoint
    llm_provider: Literal["local", "remote"] = "local"
    rerank_provider: Literal["local", "remote"] = "local"

    # Remote endpoint details
    remote_llm: RemoteEndpointConfig = Field(default_factory=RemoteEndpointConfig)
    remote_rerank: RemoteEndpointConfig = Field(default_factory=RemoteEndpointConfig)

    # Optional separate vision model for remote (e.g. gpt-4o for vision, claude for text)
    remote_vision_model: str = ""


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_path: Path | None = None
_provider_settings: ProviderSettings | None = None


def _get_config_path() -> Path:
    global _config_path
    if _config_path is None:
        _config_path = settings.paths.runtime_root / "provider_config.json"
    return _config_path


def get_provider_settings() -> ProviderSettings:
    """Return the mutable singleton.  Create it lazily on first call."""
    global _provider_settings
    if _provider_settings is None:
        _provider_settings = _load()
    return _provider_settings


def _load() -> ProviderSettings:
    path = _get_config_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ps = ProviderSettings(**data)
            logger.info("Loaded provider config from %s", path)
            return ps
        except Exception as exc:
            logger.warning("Failed to parse provider config (%s): %s — using defaults", path, exc)
    return ProviderSettings()


def save_provider_settings(ps: ProviderSettings | None = None) -> None:
    """Persist the current (or supplied) settings to disk."""
    if ps is None:
        ps = get_provider_settings()
    path = _get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ps.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved provider config to %s", path)


def update_provider_settings(patch: dict) -> ProviderSettings:
    """Merge *patch* into the current settings, persist, and return."""
    ps = get_provider_settings()

    # Handle flat fields
    for key in ("llm_provider", "rerank_provider", "remote_vision_model"):
        if key in patch:
            setattr(ps, key, patch[key])

    # Handle nested remote_llm / remote_rerank dicts
    for nested_key in ("remote_llm", "remote_rerank"):
        if nested_key in patch and isinstance(patch[nested_key], dict):
            current = getattr(ps, nested_key)
            for field, value in patch[nested_key].items():
                if hasattr(current, field):
                    setattr(current, field, value)

    save_provider_settings(ps)
    # Update the module-level singleton so everyone sees the change
    global _provider_settings
    _provider_settings = ps
    return ps


# ---------------------------------------------------------------------------
# Convenience helpers used by LlmClient / RerankClient
# ---------------------------------------------------------------------------

def is_llm_remote() -> bool:
    return get_provider_settings().llm_provider == "remote"


def is_rerank_remote() -> bool:
    return get_provider_settings().rerank_provider == "remote"


def get_remote_llm_config() -> RemoteEndpointConfig:
    return get_provider_settings().remote_llm


def get_remote_rerank_config() -> RemoteEndpointConfig:
    return get_provider_settings().remote_rerank


def get_remote_vision_model() -> str:
    """Return the vision model id, falling back to the main LLM model."""
    ps = get_provider_settings()
    return ps.remote_vision_model or ps.remote_llm.model
