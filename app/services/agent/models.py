"""Pydantic models for agent mode requests, events and tool contracts."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Request / Response ──────────────────────────────────────────────────

class AgentRequest(BaseModel):
    """Incoming request to the agent endpoint."""
    query: str
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    max_iterations: int = Field(default=10, ge=1, le=20)


class ExternalAgentRunRequest(AgentRequest):
    """Incoming request to create an externally-pollable agent run."""


AgentRunStatus = Literal["queued", "running", "completed", "failed"]


class AgentRunCreated(BaseModel):
    """Response returned when an external agent run is created."""
    run_id: str
    status: AgentRunStatus


class AgentRunEventRecord(BaseModel):
    """Persisted external event record for polling clients."""
    seq: int
    timestamp_ms: int
    event: "AgentEvent"


class AgentRunState(BaseModel):
    """Current state of an external agent run."""
    run_id: str
    status: AgentRunStatus
    query: str
    created_at_ms: int
    updated_at_ms: int
    completed_at_ms: Optional[int] = None
    current_phase: str = "queued"
    latest_message: Optional[str] = None
    final_answer: str = ""
    error: Optional[str] = None
    events_count: int = 0


class AgentRunEventsResponse(BaseModel):
    """Polled event batch for an external agent run."""
    run_id: str
    status: AgentRunStatus
    next_seq: int
    events: list[AgentRunEventRecord] = Field(default_factory=list)


# ── Tool Contract ───────────────────────────────────────────────────────

class ToolParameter(BaseModel):
    """Single parameter in a tool's JSON-Schema spec."""
    name: str
    type: str  # "string", "integer", "boolean", etc.
    description: str
    required: bool = True
    enum: Optional[list[str]] = None


class ToolSpec(BaseModel):
    """Declarative specification for one tool the agent can invoke."""
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    side_effect: bool = False  # True → requires user approval before execution

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI-style function-calling schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolCall(BaseModel):
    """A single tool invocation parsed from the LLM response."""
    id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result returned after executing a tool."""
    call_id: str
    tool_name: str
    success: bool
    output: str  # Stringified result (truncated if needed)


# ── Streaming Events ────────────────────────────────────────────────────

AgentEventType = Literal[
    "thinking_step",     # Agent reasoning / status
    "tool_call",         # Agent wants to invoke a tool
    "tool_result",       # Tool execution result
    "token",             # Final-answer token
    "status",            # Status update
    "error",             # Error message
    "done",              # Stream complete
]


class AgentEvent(BaseModel):
    """One NDJSON event emitted in the agent stream."""
    type: AgentEventType
    data: Any = None
