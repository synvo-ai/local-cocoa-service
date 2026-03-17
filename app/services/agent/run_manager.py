"""In-memory manager for externally-pollable agent runs."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from services.llm.client import LlmClient

from .models import (
    AgentEvent,
    AgentRunCreated,
    AgentRunEventRecord,
    AgentRunEventsResponse,
    AgentRunState,
    ExternalAgentRunRequest,
)
from .orchestrator import AgentOrchestrator
from .registry import ToolRegistry

logger = logging.getLogger(__name__)

_RUN_TTL_SECONDS = 3600
_MAX_COMPLETED_RUNS = 200


@dataclass
class _RunRecord:
    run_id: str
    request: ExternalAgentRunRequest
    status: str = "queued"
    current_phase: str = "queued"
    latest_message: str | None = None
    final_answer: str = ""
    error: str | None = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    completed_at_ms: int | None = None
    events: list[AgentRunEventRecord] = field(default_factory=list)
    task: asyncio.Task | None = None


class AgentRunManager:
    """Tracks externally-pollable agent runs in memory."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create_run(self, request: ExternalAgentRunRequest) -> AgentRunCreated:
        async with self._lock:
            self._cleanup_locked()
            run_id = uuid.uuid4().hex[:12]
            self._runs[run_id] = _RunRecord(run_id=run_id, request=request)
            return AgentRunCreated(run_id=run_id, status="queued")

    async def start_run(self, run_id: str, llm: LlmClient, registry: ToolRegistry) -> None:
        async with self._lock:
            record = self._require_run_locked(run_id)
            if record.task is not None:
                return
            record.status = "running"
            record.current_phase = "planning"
            record.latest_message = "Starting agent run"
            now_ms = int(time.time() * 1000)
            record.updated_at_ms = now_ms
            record.task = asyncio.create_task(self._run_task(record, llm, registry))

    async def get_run(self, run_id: str) -> AgentRunState:
        async with self._lock:
            record = self._require_run_locked(run_id)
            return self._to_state(record)

    async def get_events(self, run_id: str, *, after_seq: int = 0, limit: int = 200) -> AgentRunEventsResponse:
        async with self._lock:
            record = self._require_run_locked(run_id)
            filtered = [event for event in record.events if event.seq > after_seq]
            batch = filtered[:limit]
            next_seq = batch[-1].seq if batch else after_seq
            return AgentRunEventsResponse(
                run_id=run_id,
                status=record.status,
                next_seq=next_seq,
                events=batch,
            )

    async def _run_task(self, record: _RunRecord, llm: LlmClient, registry: ToolRegistry) -> None:
        orchestrator = AgentOrchestrator(llm, registry, approval_mode="auto_execute")

        try:
            async for event in orchestrator.run_events(record.request):
                await self._append_event(record.run_id, event)
        except Exception as exc:
            logger.exception("External agent run %s failed", record.run_id)
            await self._append_event(record.run_id, AgentEvent(type="error", data=str(exc)))
            await self._append_event(record.run_id, AgentEvent(type="done"))

    async def _append_event(self, run_id: str, event: AgentEvent) -> None:
        async with self._lock:
            record = self._require_run_locked(run_id)
            timestamp_ms = int(time.time() * 1000)
            seq = len(record.events) + 1
            record.events.append(AgentRunEventRecord(seq=seq, timestamp_ms=timestamp_ms, event=event))
            record.updated_at_ms = timestamp_ms
            self._update_state_from_event(record, event, timestamp_ms)

    def _update_state_from_event(self, record: _RunRecord, event: AgentEvent, timestamp_ms: int) -> None:
        if event.type == "status":
            status_value = str(event.data)
            record.latest_message = status_value
            if status_value == "agent_starting":
                record.current_phase = "planning"
                record.status = "running"
            elif status_value == "answering":
                record.current_phase = "answering"
                record.status = "running"
        elif event.type == "thinking_step":
            data = event.data if isinstance(event.data, dict) else {}
            record.latest_message = str(data.get("title") or "Thinking")
            step_type = str(data.get("type") or "")
            if step_type == "analyze":
                record.current_phase = "planning"
            elif step_type == "search":
                record.current_phase = "tool_execution"
            else:
                record.current_phase = "synthesizing"
        elif event.type == "tool_call":
            data = event.data if isinstance(event.data, dict) else {}
            record.current_phase = "tool_execution"
            record.latest_message = f"Calling tool: {data.get('tool', 'unknown')}"
        elif event.type == "tool_result":
            data = event.data if isinstance(event.data, dict) else {}
            record.current_phase = "tool_execution"
            record.latest_message = f"Completed tool: {data.get('tool', 'unknown')}"
        elif event.type == "token":
            record.current_phase = "answering"
            record.status = "running"
            record.final_answer += str(event.data or "")
        elif event.type == "error":
            record.status = "failed"
            record.current_phase = "failed"
            record.error = str(event.data)
            record.latest_message = record.error
            record.completed_at_ms = timestamp_ms
        elif event.type == "done":
            if record.status != "failed":
                record.status = "completed"
                record.current_phase = "completed"
                record.latest_message = record.latest_message or "Completed"
                record.completed_at_ms = timestamp_ms

    def _cleanup_locked(self) -> None:
        now_ms = int(time.time() * 1000)
        expired: list[str] = []
        completed: list[tuple[int, str]] = []

        for run_id, record in self._runs.items():
            completed_at = record.completed_at_ms or record.updated_at_ms
            if now_ms - completed_at > _RUN_TTL_SECONDS * 1000:
                expired.append(run_id)
            elif record.status in {"completed", "failed"}:
                completed.append((completed_at, run_id))

        for run_id in expired:
            del self._runs[run_id]

        if len(completed) > _MAX_COMPLETED_RUNS:
            completed.sort()
            for _, run_id in completed[:-_MAX_COMPLETED_RUNS]:
                self._runs.pop(run_id, None)

    def _require_run_locked(self, run_id: str) -> _RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    @staticmethod
    def _to_state(record: _RunRecord) -> AgentRunState:
        return AgentRunState(
            run_id=record.run_id,
            status=record.status,
            query=record.request.query,
            created_at_ms=record.created_at_ms,
            updated_at_ms=record.updated_at_ms,
            completed_at_ms=record.completed_at_ms,
            current_phase=record.current_phase,
            latest_message=record.latest_message,
            final_answer=record.final_answer,
            error=record.error,
            events_count=len(record.events),
        )
