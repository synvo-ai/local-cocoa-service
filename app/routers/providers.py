"""
Providers router — manage local ↔ remote switching for LLM / Reranker.
"""

from __future__ import annotations

import httpx
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.model_manager import get_model_manager, ModelType
from core.provider_config import (
    get_provider_settings,
    update_provider_settings,
    ProviderSettings,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/providers", tags=["providers"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ProviderPatch(BaseModel):
    """Partial update for provider settings."""
    llm_provider: str | None = None
    rerank_provider: str | None = None
    remote_llm: dict[str, Any] | None = None
    remote_rerank: dict[str, Any] | None = None
    remote_vision_model: str | None = None


class TestConnectionRequest(BaseModel):
    base_url: str
    api_key: str = ""
    model: str = ""
    provider_hint: str = ""


class TestConnectionResponse(BaseModel):
    ok: bool
    latency_ms: float | None = None
    model_echo: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Return current provider settings."""
    ps = get_provider_settings()
    return ps.model_dump()


@router.put("/config")
async def put_config(patch: ProviderPatch) -> dict[str, Any]:
    """Update provider settings.  Only supplied fields are changed."""
    raw = patch.model_dump(exclude_none=True)
    if not raw:
        raise HTTPException(status_code=422, detail="No fields to update")

    old_ps = get_provider_settings()
    old_llm = old_ps.llm_provider
    old_rerank = old_ps.rerank_provider

    ps = update_provider_settings(raw)

    # Handle model process lifecycle on provider switch
    manager = get_model_manager()
    if ps.llm_provider != old_llm:
        if ps.llm_provider == "remote":
            manager.stop_model(ModelType.VISION)
    if ps.rerank_provider != old_rerank:
        if ps.rerank_provider == "remote":
            manager.stop_model(ModelType.RERANK)

    logger.info("Provider config updated: llm=%s, rerank=%s", ps.llm_provider, ps.rerank_provider)
    return ps.model_dump()


@router.post("/test-connection", response_model=TestConnectionResponse)
async def test_connection(req: TestConnectionRequest) -> TestConnectionResponse:
    """
    Fire a tiny request at the given endpoint to verify reachability,
    auth, and model availability.  Returns latency and echoed model id.
    """
    base = req.base_url.rstrip("/")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if req.api_key:
        headers["Authorization"] = f"Bearer {req.api_key}"

    tok_key = "max_completion_tokens" if req.provider_hint == "openai" else "max_tokens"
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Hi"}],
        tok_key: 50,
        "stream": False,
    }
    if req.model:
        payload["model"] = req.model

    endpoint = f"{base}/v1/chat/completions"
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            latency = round((time.monotonic() - t0) * 1000, 1)

            if response.status_code == 401:
                return TestConnectionResponse(ok=False, latency_ms=latency, error="Authentication failed (401). Check your API key.")
            if response.status_code == 404:
                return TestConnectionResponse(ok=False, latency_ms=latency, error=f"Endpoint not found (404). Tried: {endpoint}")

            response.raise_for_status()
            data = response.json()
            model_echo = data.get("model", "")
            return TestConnectionResponse(ok=True, latency_ms=latency, model_echo=model_echo or req.model)

    except httpx.ConnectError:
        return TestConnectionResponse(ok=False, error=f"Cannot connect to {base}. Check the URL.")
    except httpx.TimeoutException:
        return TestConnectionResponse(ok=False, error="Connection timed out (15 s).")
    except httpx.HTTPStatusError as exc:
        latency = round((time.monotonic() - t0) * 1000, 1)
        body = (exc.response.text or "")[:300]
        return TestConnectionResponse(ok=False, latency_ms=latency, error=f"HTTP {exc.response.status_code}: {body}")
    except Exception as exc:
        return TestConnectionResponse(ok=False, error=f"{type(exc).__name__}: {exc}")
