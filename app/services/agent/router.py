"""FastAPI router for agent mode – ``POST /agent/stream``."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.context import get_llm_client

from .executor import TOOL_EXECUTORS
from .models import AgentRequest
from .orchestrator import AgentOrchestrator
from .registry import ToolRegistry, pop_pending_action

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])

# ── Singleton registry, initialised once ────────────────────────────────

_registry: ToolRegistry | None = None


def _get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        for name, fn in TOOL_EXECUTORS.items():
            _registry.register(name, fn)
    return _registry


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/agent/stream")
async def agent_stream(payload: AgentRequest) -> StreamingResponse:
    """Stream an agent-mode conversation.

    The response is NDJSON with ``AgentEvent`` objects:
    ``thinking_step``, ``tool_call``, ``tool_result``, ``token``, ``status``,
    ``error``, ``done``.
    """
    logger.info("POST /agent/stream: query=%s", payload.query[:120])

    llm = get_llm_client()
    registry = _get_registry()
    orchestrator = AgentOrchestrator(llm, registry)

    return StreamingResponse(
        orchestrator.run(payload),
        media_type="application/x-ndjson",
    )


# ── Tool confirmation ───────────────────────────────────────────────────

class ToolConfirmRequest(BaseModel):
    confirmation_id: str
    overrides: dict[str, Any] | None = None


@router.post("/agent/tool/confirm")
async def confirm_tool(payload: ToolConfirmRequest):
    """Execute a previously-pending side-effect tool after user approval."""
    action = pop_pending_action(payload.confirmation_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Confirmation expired or not found.")

    tool_name = action["tool_name"]
    arguments = dict(action["arguments"])

    if payload.overrides:
        if tool_name == "send_email":
            allowed = {"account_id", "to", "subject", "body"}
            for key, value in payload.overrides.items():
                if key not in allowed:
                    continue
                if value is None:
                    continue
                arguments[key] = str(value)
        else:
            arguments.update(payload.overrides)

    executor = TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        raise HTTPException(status_code=400, detail=f"Tool '{tool_name}' not available.")

    try:
        result = await executor(arguments)
        return {"status": "executed", "tool": tool_name, "result": result}
    except Exception as exc:
        logger.error("Confirmed tool '%s' failed: %s", tool_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/agent/tool/cancel")
async def cancel_tool(payload: ToolConfirmRequest):
    """Cancel a pending side-effect tool."""
    action = pop_pending_action(payload.confirmation_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Confirmation expired or not found.")
    return {"status": "cancelled", "tool": action["tool_name"]}
