"""In-memory manager for externally-pollable agent runs."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from services.llm.client import LlmClient

from .models import (
    AgentEvent,
    AgentExecutionState,
    AgentRunCreated,
    AgentRunEventRecord,
    AgentRunEventsResponse,
    AgentRunState,
    ExternalAgentRunRequest,
    PendingRunAction,
    ToolCall,
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
    execution_state: AgentExecutionState
    approval_mode: str
    status: str = "queued"
    current_phase: str = "queued"
    latest_message: str | None = None
    final_answer: str = ""
    error: str | None = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    completed_at_ms: int | None = None
    events: list[AgentRunEventRecord] = field(default_factory=list)
    pending_action: PendingRunAction | None = None
    tool_calls: dict[str, dict[str, object]] = field(default_factory=dict)
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
            self._runs[run_id] = _RunRecord(
                run_id=run_id,
                request=request,
                execution_state=AgentExecutionState(messages=[], next_iteration=0, max_iterations=request.max_iterations),
                approval_mode=request.approval_mode,
            )
            return AgentRunCreated(run_id=run_id, status="queued")

    async def start_run(self, run_id: str, llm: LlmClient, registry: ToolRegistry) -> None:
        async with self._lock:
            record = self._require_run_locked(run_id)
            if record.task is not None:
                return
            if record.pending_action is not None:
                return
            if not record.execution_state.messages:
                orchestrator = AgentOrchestrator(llm, registry, approval_mode=self._orchestrator_approval_mode(record.approval_mode))
                record.execution_state = orchestrator.create_execution_state(record.request)
            record.status = "running"
            if record.current_phase == "queued":
                record.current_phase = "planning"
            record.latest_message = "Starting agent run" if record.execution_state.next_iteration == 0 else "Resuming agent run"
            now_ms = int(time.time() * 1000)
            record.updated_at_ms = now_ms
            emit_start = record.execution_state.next_iteration == 0
            record.task = asyncio.create_task(self._run_task(record.run_id, llm, registry, emit_start=emit_start))

    async def confirm_run(self, run_id: str, confirmation_id: str, overrides: dict[str, object] | None, llm: LlmClient, registry: ToolRegistry) -> AgentRunState:
        async with self._lock:
            record = self._require_run_locked(run_id)
            pending = self._require_pending_action(record, confirmation_id)
            merged_args = dict(pending.args)
            if overrides:
                for key, value in overrides.items():
                    if key not in pending.editable_fields or value is None:
                        continue
                    merged_args[key] = str(value)

            record.pending_action = None
            record.status = "running"
            record.current_phase = "tool_execution"
            record.latest_message = f"Executing confirmed tool: {pending.tool}"
            record.updated_at_ms = int(time.time() * 1000)

        result = await registry.execute(
            ToolCall(id=pending.call_id, tool_name=pending.tool, arguments=merged_args),
            approval_mode="auto_execute",
        )
        await self._append_confirmed_tool_result(run_id, pending, merged_args, result)
        await self.start_run(run_id, llm, registry)
        return await self.get_run(run_id)

    async def cancel_run(self, run_id: str, confirmation_id: str, llm: LlmClient, registry: ToolRegistry) -> AgentRunState:
        async with self._lock:
            record = self._require_run_locked(run_id)
            pending = self._require_pending_action(record, confirmation_id)
            record.pending_action = None
            record.status = "running"
            record.current_phase = "tool_execution"
            record.latest_message = f"Cancelled tool: {pending.tool}"
            record.updated_at_ms = int(time.time() * 1000)

        cancelled_output = {
            "status": "cancelled",
            "tool": pending.tool,
            "message": f"User cancelled tool '{pending.tool}'.",
        }
        await self._append_cancelled_tool_result(run_id, pending, cancelled_output)
        await self.start_run(run_id, llm, registry)
        return await self.get_run(run_id)

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

    async def _run_task(self, run_id: str, llm: LlmClient, registry: ToolRegistry, *, emit_start: bool) -> None:
        async with self._lock:
            record = self._require_run_locked(run_id)
            approval_mode = self._orchestrator_approval_mode(record.approval_mode)
            execution_state = record.execution_state

        orchestrator = AgentOrchestrator(llm, registry, approval_mode=approval_mode)

        try:
            async for event in orchestrator.run_state_events(execution_state, emit_start=emit_start):
                await self._append_event(run_id, event)
        except Exception as exc:
            logger.exception("External agent run %s failed", run_id)
            await self._append_event(run_id, AgentEvent(type="error", data=str(exc)))
            await self._append_event(run_id, AgentEvent(type="done"))
        finally:
            async with self._lock:
                record = self._runs.get(run_id)
                if record is not None:
                    record.task = None

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
            call_id = str(data.get("call_id") or "")
            if call_id:
                record.tool_calls[call_id] = data
        elif event.type == "tool_result":
            data = event.data if isinstance(event.data, dict) else {}
            tool_name = str(data.get("tool", "unknown"))
            if data.get("requires_confirmation"):
                call_id = str(data.get("call_id") or "")
                tool_call = record.tool_calls.get(call_id, {})
                confirmation_id = str(data.get("confirmation_id") or uuid.uuid4().hex[:12])
                message = str(data.get("confirmation_message") or "This action requires confirmation before it can be executed.")
                args = dict(tool_call.get("args") or {})
                record.pending_action = PendingRunAction(
                    confirmation_id=confirmation_id,
                    call_id=call_id,
                    tool=tool_name,
                    message=message,
                    args=args,
                    editable_fields=self._editable_fields_for_tool(tool_name),
                )
                record.status = "awaiting_confirmation"
                record.current_phase = "awaiting_confirmation"
                record.latest_message = message
                record.completed_at_ms = None
            else:
                record.current_phase = "tool_execution"
                record.latest_message = f"Completed tool: {tool_name}"
                call_id = str(data.get("call_id") or "")
                if call_id:
                    record.tool_calls.pop(call_id, None)
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
            if record.pending_action is not None:
                record.status = "awaiting_confirmation"
                record.current_phase = "awaiting_confirmation"
            elif record.status != "failed":
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
    def _orchestrator_approval_mode(approval_mode: str) -> str:
        if approval_mode == "require_confirmation":
            return "run_confirmation"
        return approval_mode

    @staticmethod
    def _editable_fields_for_tool(tool_name: str) -> list[str]:
        if tool_name == "send_email":
            return ["account_id", "to", "subject", "body"]
        if tool_name == "create_note":
            return ["title", "body"]
        return []

    @staticmethod
    def _require_pending_action(record: _RunRecord, confirmation_id: str) -> PendingRunAction:
        pending = record.pending_action
        if pending is None or pending.confirmation_id != confirmation_id:
            raise KeyError(confirmation_id)
        return pending

    async def _append_confirmed_tool_result(
        self,
        run_id: str,
        pending: PendingRunAction,
        merged_args: dict[str, object],
        result,
    ) -> None:
        event = AgentEvent(type="tool_result", data={
            "call_id": result.call_id,
            "tool": result.tool_name,
            "success": result.success,
            "output_preview": result.output[:500],
            "requires_confirmation": False,
            "confirmation_id": pending.confirmation_id,
            "confirmation_message": None,
        })
        await self._append_event(run_id, event)
        async with self._lock:
            record = self._require_run_locked(run_id)
            record.execution_state.messages.append({
                "role": "tool",
                "tool_call_id": pending.call_id,
                "content": result.output,
            })
            record.tool_calls.pop(pending.call_id, None)

    async def _append_cancelled_tool_result(
        self,
        run_id: str,
        pending: PendingRunAction,
        cancelled_output: dict[str, str],
    ) -> None:
        output = json.dumps(cancelled_output)
        event = AgentEvent(type="tool_result", data={
            "call_id": pending.call_id,
            "tool": pending.tool,
            "success": False,
            "output_preview": output[:500],
            "requires_confirmation": False,
            "confirmation_id": pending.confirmation_id,
            "confirmation_message": None,
        })
        await self._append_event(run_id, event)
        async with self._lock:
            record = self._require_run_locked(run_id)
            record.execution_state.messages.append({
                "role": "tool",
                "tool_call_id": pending.call_id,
                "content": output,
            })
            record.tool_calls.pop(pending.call_id, None)

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
            approval_mode=record.approval_mode,
            pending_action=record.pending_action,
        )
