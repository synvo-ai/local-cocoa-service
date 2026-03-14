"""Tool registry – maps tool specs to their executor callables."""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable

from .models import ToolCall, ToolResult, ToolSpec
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# Executor callable signature: (arguments: dict) -> str
ToolExecutorFn = Callable[[dict[str, Any]], Awaitable[str]]

# In-memory store for pending confirmations (confirmation_id -> details)
_pending_actions: dict[str, dict[str, Any]] = {}
_PENDING_TTL = 300  # 5 minutes


def _cleanup_expired() -> None:
    now = time.time()
    expired = [k for k, v in _pending_actions.items() if now - v["created_at"] > _PENDING_TTL]
    for k in expired:
        del _pending_actions[k]


def store_pending_action(tool_name: str, arguments: dict[str, Any]) -> str:
    """Store a side-effect tool call for later confirmation. Returns confirmation_id."""
    _cleanup_expired()
    cid = uuid.uuid4().hex[:12]
    _pending_actions[cid] = {
        "tool_name": tool_name,
        "arguments": arguments,
        "created_at": time.time(),
    }
    return cid


def pop_pending_action(confirmation_id: str) -> dict[str, Any] | None:
    """Retrieve and remove a pending action. Returns None if expired/missing."""
    _cleanup_expired()
    return _pending_actions.pop(confirmation_id, None)


class ToolRegistry:
    """Holds the fixed set of tools and their executors.

    Usage::

        registry = ToolRegistry()
        registry.register("workspace_search", my_search_fn)
        result = await registry.execute(tool_call)
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {t.name: t for t in ALL_TOOLS}
        self._executors: dict[str, ToolExecutorFn] = {}

    # ── Registration ────────────────────────────────────────────────────

    def register(self, tool_name: str, executor: ToolExecutorFn) -> None:
        if tool_name not in self._specs:
            raise ValueError(f"Unknown tool: {tool_name}")
        self._executors[tool_name] = executor
        logger.info("Registered executor for tool '%s'", tool_name)

    # ── Queries ─────────────────────────────────────────────────────────

    @property
    def specs(self) -> list[ToolSpec]:
        """Return specs only for tools that have registered executors."""
        return [s for s in self._specs.values() if s.name in self._executors]

    def openai_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schema for all registered tools."""
        return [s.to_openai_function() for s in self.specs]

    def get_spec(self, tool_name: str) -> ToolSpec | None:
        return self._specs.get(tool_name)

    # ── Execution ───────────────────────────────────────────────────────

    async def execute(self, call: ToolCall, *, max_output_chars: int = 8000) -> ToolResult:
        """Execute a tool call and return a bounded result.

        Side-effect tools (``side_effect=True``) are NOT executed immediately.
        Instead a pending confirmation is created and the result instructs the
        agent (and frontend) to ask the user for approval.
        """
        executor = self._executors.get(call.tool_name)
        if executor is None:
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                output=f"Tool '{call.tool_name}' is not available.",
            )

        # Gate side-effect tools behind confirmation
        spec = self.get_spec(call.tool_name)
        if spec and spec.side_effect:
            cid = store_pending_action(call.tool_name, call.arguments)
            pending_result = json.dumps({
                "status": "pending_confirmation",
                "confirmation_id": cid,
                "tool": call.tool_name,
                "message": f"This action requires your confirmation before it can be executed.",
            })
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=True,
                output=pending_result,
            )

        try:
            raw = await executor(call.arguments)
            # Truncate if too long to avoid context overflow
            if len(raw) > max_output_chars:
                raw = raw[:max_output_chars] + "\n...[truncated]"
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=True,
                output=raw,
            )
        except Exception as exc:
            logger.warning("Tool '%s' failed: %s", call.tool_name, exc, exc_info=True)
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                output=f"Error executing {call.tool_name}: {exc}",
            )
