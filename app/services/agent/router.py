"""FastAPI router for agent mode – UI stream + external pollable runs."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.context import get_llm_client

from .executor import TOOL_EXECUTORS
from .models import (
    AgentRequest,
    AgentRunCreated,
    AgentRunEventsResponse,
    AgentRunState,
    ExternalRunCancelRequest,
    ExternalRunConfirmRequest,
    ExternalAgentRunRequest,
)
from .orchestrator import AgentOrchestrator
from .run_manager import AgentRunManager
from .registry import ToolRegistry, pop_pending_action

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])

# ── Singleton registry, initialised once ────────────────────────────────

_registry: ToolRegistry | None = None
_run_manager: AgentRunManager | None = None


def _get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        for name, fn in TOOL_EXECUTORS.items():
            _registry.register(name, fn)
    return _registry


def _get_run_manager() -> AgentRunManager:
    global _run_manager
    if _run_manager is None:
        _run_manager = AgentRunManager()
    return _run_manager


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
    orchestrator = AgentOrchestrator(llm, registry, approval_mode="require_confirmation")

    return StreamingResponse(
        orchestrator.run(payload),
        media_type="application/x-ndjson",
    )


@router.post("/agent/external/runs", response_model=AgentRunCreated)
async def create_external_run(payload: ExternalAgentRunRequest) -> AgentRunCreated:
    """Create an externally-pollable agent run."""
    logger.info("POST /agent/external/runs: query=%s", payload.query[:120])

    llm = get_llm_client()
    registry = _get_registry()
    run_manager = _get_run_manager()

    created = await run_manager.create_run(payload)
    await run_manager.start_run(created.run_id, llm, registry)
    return created


@router.get("/agent/external/runs/{run_id}", response_model=AgentRunState)
async def get_external_run(run_id: str) -> AgentRunState:
    """Get the latest state for an external agent run."""
    run_manager = _get_run_manager()
    try:
        return await run_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


@router.get("/agent/external/runs/{run_id}/events", response_model=AgentRunEventsResponse)
async def get_external_run_events(
    run_id: str,
    after_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> AgentRunEventsResponse:
    """Poll structured events for an external agent run."""
    run_manager = _get_run_manager()
    try:
        return await run_manager.get_events(run_id, after_seq=after_seq, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


@router.post("/agent/external/runs/{run_id}/confirm", response_model=AgentRunState)
async def confirm_external_run(run_id: str, payload: ExternalRunConfirmRequest) -> AgentRunState:
    """Confirm a pending run-scoped action, optionally editing fields first."""
    llm = get_llm_client()
    registry = _get_registry()
    run_manager = _get_run_manager()
    try:
        return await run_manager.confirm_run(run_id, payload.confirmation_id, payload.overrides, llm, registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Pending action not found.") from exc


@router.post("/agent/external/runs/{run_id}/cancel", response_model=AgentRunState)
async def cancel_external_run(run_id: str, payload: ExternalRunCancelRequest) -> AgentRunState:
    """Cancel a pending run-scoped action and continue the run."""
    llm = get_llm_client()
    registry = _get_registry()
    run_manager = _get_run_manager()
    try:
        return await run_manager.cancel_run(run_id, payload.confirmation_id, llm, registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Pending action not found.") from exc


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
