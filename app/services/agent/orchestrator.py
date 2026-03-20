"""Agent orchestrator - the core ReAct loop.

Uses prompt-based tool calling with <tool_call> tags for maximum compatibility
with local models (llama.cpp) that do not support native function calling.

The orchestrator yields NDJSON ``AgentEvent`` lines suitable for streaming
back through a ``StreamingResponse``.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator

from services.llm.client import LlmClient
from core.provider_config import get_provider_settings, is_llm_remote

from .models import AgentEvent, AgentExecutionState, AgentRequest, ToolCall, ToolResult
from .prompts import build_system_prompt, build_system_prompt_with_fallback
from .registry import ToolRegistry

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 10
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _parse_pending_confirmation(output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and parsed.get("status") in {"pending_confirmation", "pending_run_confirmation"}:
        return parsed
    return None


def _emit(event: AgentEvent) -> str:
    """Serialize one event as an NDJSON line."""
    return event.model_dump_json() + "\n"


def _thinking_event(title: str, *, summary: str = "", step_type: str = "info", elapsed_ms: int = 0) -> AgentEvent:
    return AgentEvent(
        type="thinking_step",
        data={
            "id": uuid.uuid4().hex[:8],
            "type": step_type,
            "title": title,
            "summary": summary,
            "status": "complete",
            "timestamp_ms": elapsed_ms,
        },
    )


class AgentOrchestrator:
    """Runs the agent loop: LLM -> tool calls -> tool results -> repeat."""

    def __init__(self, llm: LlmClient, registry: ToolRegistry, *, approval_mode: str = "require_confirmation") -> None:
        self.llm = llm
        self.registry = registry
        self.approval_mode = approval_mode

    def _is_native_mode(self) -> bool:
        """Return True when the active provider supports native tool calling.

        Native mode is used for:
        - Remote OpenAI / Anthropic / vLLM providers
        - Local llama-server (started with --jinja, supports OpenAI-style tools)
        """
        if is_llm_remote():
            hint = get_provider_settings().remote_llm.provider_hint or "openai"
            return hint in ("openai", "anthropic", "vllm")
        # Local llama-server with --jinja supports native OpenAI-style tool calling
        return True

    def _is_gemini_native_mode(self) -> bool:
        """Return True when Gemini native function calling should be used."""
        if not is_llm_remote():
            return False
        remote = get_provider_settings().remote_llm
        base_url = remote.base_url or ""
        hint = remote.provider_hint or ""
        return hint == "gemini" or "generativelanguage.googleapis.com" in base_url

    def _provider_hint(self) -> str:
        if is_llm_remote():
            return get_provider_settings().remote_llm.provider_hint or "openai"
        return "local"

    def _uses_native_prompt(self) -> bool:
        return self._is_native_mode() or self._is_gemini_native_mode()

    def create_execution_state(self, request: AgentRequest) -> AgentExecutionState:
        """Create resumable execution state for a new request."""
        messages: list[dict[str, Any]] = []
        if self._uses_native_prompt():
            system_prompt = build_system_prompt(self.registry.specs, approval_mode=self.approval_mode)
        else:
            system_prompt = build_system_prompt_with_fallback(self.registry.specs, approval_mode=self.approval_mode)
        messages.append({"role": "system", "content": system_prompt})

        for msg in request.conversation_history[-10:]:
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": str(msg.get("content", ""))})

        messages.append({"role": "user", "content": request.query})
        return AgentExecutionState(messages=messages, next_iteration=0, max_iterations=request.max_iterations)

    async def run(self, request: AgentRequest) -> AsyncGenerator[str, None]:
        """Execute the agent loop and yield NDJSON events."""
        async for event in self.run_events(request):
            yield _emit(event)

    async def run_events(self, request: AgentRequest) -> AsyncGenerator[AgentEvent, None]:
        """Execute the agent loop and yield structured events."""
        state = self.create_execution_state(request)
        async for event in self.run_state_events(state, emit_start=True):
            yield event

    async def run_state_events(
        self,
        state: AgentExecutionState,
        *,
        emit_start: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Resume the agent loop from previously saved execution state."""
        t0 = time.perf_counter()
        messages = state.messages

        if emit_start:
            yield AgentEvent(type="status", data="agent_starting")

        def _elapsed() -> int:
            return int((time.perf_counter() - t0) * 1000)

        native_mode = self._is_native_mode()
        gemini_native_mode = self._is_gemini_native_mode()
        tool_calling_mode = "native" if (native_mode or gemini_native_mode) else "fallback"

        # Build provider info for the UI
        if is_llm_remote():
            ps = get_provider_settings()
            provider_type = "cloud"
            model_label = ps.remote_llm.model or ps.remote_llm.provider_hint or "remote"
        else:
            from core.config import settings as _settings
            provider_type = "local"
            # Derive a short label from the VLM model filename
            vlm_file = _settings.paths.vlm_model
            model_label = vlm_file.rsplit("/", 1)[-1].rsplit(".", 1)[0] if vlm_file else "local model"

        if emit_start:
            yield _thinking_event("Analyzing your request...", step_type="analyze", elapsed_ms=_elapsed())

        # Tell the UI which tool-calling mode and provider are active
        yield AgentEvent(type="status", data={
            "tool_calling_mode": tool_calling_mode,
            "provider_type": provider_type,
            "model_label": model_label,
        })

        for iteration in range(state.next_iteration, state.max_iterations):
            logger.info("Agent iteration %d/%d (mode=%s)", iteration + 1, state.max_iterations, tool_calling_mode)

            # -- Call LLM (Gemini native / general native / prompt fallback) --
            try:
                if gemini_native_mode:
                    tool_calls, assistant_text = await self._call_gemini_native(messages)
                elif native_mode:
                    provider_hint = self._provider_hint()
                    tool_calls, assistant_text = await self._call_native(messages, provider_hint)
                else:
                    tool_calls, assistant_text = await self._call_fallback(messages)
            except Exception as exc:
                if native_mode or gemini_native_mode:
                    logger.warning("Native tool call failed (%s), falling back to XML mode", exc)
                    try:
                        tool_calls, assistant_text = await self._call_fallback(messages)
                        tool_calling_mode = "fallback"
                    except Exception as fallback_exc:
                        logger.error("Fallback also failed: %s", fallback_exc)
                        yield AgentEvent(type="error", data=str(fallback_exc))
                        yield AgentEvent(type="done")
                        return
                else:
                    logger.error("Agent LLM call failed: %s", exc)
                    yield AgentEvent(type="error", data=str(exc))
                    yield AgentEvent(type="done")
                    return

            # -- No tool calls -> final answer ----------------------------
            if not tool_calls:
                if iteration < state.max_iterations - 1 and self._should_retry_for_missed_send_email(messages, assistant_text):
                    logger.info("Retrying agent turn because model narrated send_email without a tool call")
                    if assistant_text:
                        messages.append({"role": "assistant", "content": assistant_text})
                    if native_mode:
                        nudge = (
                            "You indicated that you are sending an email. "
                            "Do not describe the action in prose. "
                            "If you have enough information, call the send_email tool now. "
                            "If information is missing, reply briefly with exactly what is missing."
                        )
                    else:
                        nudge = (
                            "You indicated that you are sending an email. "
                            "Do not describe the action in prose. "
                            "If you have enough information, respond with ONLY a <tool_call> block for send_email. "
                            "If information is missing, reply briefly with exactly what is missing."
                        )
                    messages.append({"role": "user", "content": nudge})
                    continue

                yield AgentEvent(type="status", data="answering")
                if assistant_text:
                    yield AgentEvent(type="token", data=assistant_text)
                yield _thinking_event("Done", step_type="synthesize", elapsed_ms=_elapsed())
                yield AgentEvent(type="done")
                return

            # -- Execute tool calls ---------------------------------------
            messages.append({"role": "assistant", "content": assistant_text or None, "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments)}}
                for tc in tool_calls
            ]})

            has_pending = False
            for tc in tool_calls:
                yield _thinking_event(
                    f"Using {tc.tool_name}...",
                    summary=json.dumps(tc.arguments, ensure_ascii=False)[:200],
                    step_type="search",
                    elapsed_ms=_elapsed(),
                )
                yield AgentEvent(type="tool_call", data={
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "call_id": tc.id,
                })

                result: ToolResult = await self.registry.execute(tc, approval_mode=self.approval_mode)
                pending_confirmation = _parse_pending_confirmation(result.output)
                if pending_confirmation is not None:
                    has_pending = True

                yield AgentEvent(type="tool_result", data={
                    "call_id": result.call_id,
                    "tool": result.tool_name,
                    "success": result.success,
                    "output_preview": result.output[:500],
                    "requires_confirmation": pending_confirmation is not None,
                    "confirmation_id": pending_confirmation.get("confirmation_id") if pending_confirmation else None,
                    "confirmation_message": pending_confirmation.get("message") if pending_confirmation else None,
                })

                if pending_confirmation is None:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.output,
                    })

            state.next_iteration = iteration + 1

            if has_pending:
                yield _thinking_event("Waiting for your confirmation...", step_type="synthesize", elapsed_ms=_elapsed())
                yield AgentEvent(type="done")
                return

            # -- Detect repeated identical calls (loop guard) -------------
            if iteration >= 2 and self._is_repeating(messages):
                logger.warning("Detected repeating tool calls, forcing final answer")
                yield _thinking_event("Preparing final answer...", step_type="synthesize", elapsed_ms=_elapsed())
                break

        # Max iterations reached or loop broken -> stream a final answer
        messages.append({
            "role": "user",
            "content": "Please provide your final answer now based on all the information you have gathered.",
        })
        yield AgentEvent(type="status", data="answering")
        async for chunk in self._stream_final_answer(messages):
            yield AgentEvent(type="token", data=chunk)
        yield _thinking_event("Done", step_type="synthesize", elapsed_ms=_elapsed())
        yield AgentEvent(type="done")

    # -- LLM call strategies ----------------------------------------------


    async def _call_native(
        self,
        messages: list[dict[str, Any]],
        provider_hint: str
    ) -> tuple[list[ToolCall], str]:
        """Parse tool calls using native tool configuration schemas."""
        tools = []
        if provider_hint == "anthropic":
            for spec in self.registry.specs:
                openai_fn = spec.to_openai_function()
                tools.append({
                    "name": openai_fn["function"]["name"],
                    "description": openai_fn["function"]["description"][:1024],
                    "input_schema": openai_fn["function"]["parameters"],
                })
        else:
            # OpenAI, vLLM, and local llama-server all use OpenAI-style tool schemas
            for spec in self.registry.specs:
                tools.append(spec.to_openai_function())

        formatted_messages = self._format_messages_for_native(messages, provider_hint)

        response = await self.llm.chat_complete_with_tools(
            formatted_messages, tools=tools, max_tokens=1024, temperature=0.1
        )

        calls: list[ToolCall] = []
        assistant_text = ""

        if provider_hint == "anthropic" and isinstance(response, list):
            for block in response:
                if block.get("type") == "text":
                    assistant_text += block.get("text", "")
                elif block.get("type") == "tool_use":
                    calls.append(ToolCall(
                        id=block.get("id", uuid.uuid4().hex[:8]),
                        tool_name=block.get("name", ""),
                        arguments=block.get("input", {})
                    ))
        elif isinstance(response, dict):
            assistant_text = response.get("content") or ""
            for raw_tc in response.get("tool_calls", []):
                # OpenAI format: {id, type, function: {name, arguments}}
                # llama-server format: {name, arguments} (no wrapper)
                fn = raw_tc.get("function")
                if fn:
                    # OpenAI / vLLM style
                    tc_name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    tc_id = raw_tc.get("id", uuid.uuid4().hex[:8])
                else:
                    # llama-server style (flat name + arguments)
                    tc_name = raw_tc.get("name", "")
                    raw_args = raw_tc.get("arguments", "{}")
                    tc_id = raw_tc.get("id", uuid.uuid4().hex[:8])
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(
                    id=tc_id,
                    tool_name=tc_name,
                    arguments=args
                ))

        return calls, assistant_text

    @staticmethod
    def _format_messages_for_native(messages: list[dict[str, Any]], provider_hint: str) -> list[dict[str, Any]]:
        clean: list[dict[str, Any]] = []
        sys_texts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            if role == "system":
                sys_texts.append(str(content))
                continue

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if provider_hint == "anthropic":
                    clean.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": str(content)
                        }]
                    })
                else: 
                    clean.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": str(content)
                    })
            else:
                if "tool_calls" in msg and provider_hint == "anthropic":
                    c_blocks = []
                    if content:
                        c_blocks.append({"type": "text", "text": str(content)})
                    for tc in msg["tool_calls"]:
                        try:
                            # Sometimes arguments is string from stringify
                            args = json.loads(tc["function"]["arguments"])
                        except (json.JSONDecodeError, TypeError, KeyError):
                            args = tc.get("function", {}).get("arguments", {})
                        c_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": tc["function"]["name"],
                            "input": args
                        })
                    clean.append({"role": "assistant", "content": c_blocks})
                else:
                    clean.append({"role": role, "content": msg.get("content", ""), **({"tool_calls": msg["tool_calls"]} if "tool_calls" in msg else {})})

        if sys_texts:
            clean.insert(0, {"role": "system", "content": "\n\n".join(sys_texts)})
            
        return clean

    async def _call_fallback(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[ToolCall], str]:
        """Parse tool calls from a plain-text response using <tool_call> tags."""
        clean_messages = self._clean_messages_for_api(messages)

        text = await self.llm.chat_complete(clean_messages, max_tokens=4096, temperature=0.1)

        matches = _TOOL_CALL_RE.findall(text)
        calls: list[ToolCall] = []
        for raw in matches:
            try:
                parsed = json.loads(raw)
                calls.append(ToolCall(
                    id=uuid.uuid4().hex[:8],
                    tool_name=parsed.get("name", ""),
                    arguments=parsed.get("arguments", {}),
                ))
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse fallback tool call: %s", raw[:200])

        # Strip tool_call tags from the text to get the "thinking" text
        clean_text = _TOOL_CALL_RE.sub("", text).strip()
        return calls, clean_text

    async def _call_gemini_native(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[ToolCall], str]:
        """Use Google Gemini SDK for native function calling."""
        from core.provider_config import get_remote_llm_config
        from services.llm.gemini_native import GeminiNativeClient

        config = get_remote_llm_config()
        client = GeminiNativeClient(api_key=config.api_key, model=config.model)

        # Pass raw messages — gemini_native handles role conversion itself
        raw_calls, text = await client.call_with_tools(
            messages, self.registry.specs,
            max_tokens=4096, temperature=0.1,
        )

        calls = [
            ToolCall(id=c["id"], tool_name=c["tool_name"], arguments=c["arguments"])
            for c in raw_calls
        ]
        return calls, text

    async def _stream_final_answer(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream the final answer using the chat completion streaming API."""
        if self._is_native_mode():
            if is_llm_remote():
                ps = get_provider_settings()
                hint = ps.remote_llm.provider_hint or "openai"
            else:
                hint = "local"
            clean_messages = self._format_messages_for_native(messages, hint)
        else:
            clean_messages = self._clean_messages_for_api(messages)

        async for chunk in self.llm.stream_chat_complete(
            clean_messages, max_tokens=4096, temperature=0.3,
        ):
            yield chunk

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _clean_messages_for_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal message format to a clean format for the LLM API.

        - Removes ``tool_calls`` from assistant messages (not all backends accept them).
        - Converts ``tool`` role messages to ``user`` role with a prefix, for
          backends that don't support the ``tool`` role.
        """
        clean: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                clean.append({
                    "role": "user",
                    "content": f"[Tool Result for {tool_id}]\n{msg.get('content', '')}",
                })
            elif role == "assistant" and "tool_calls" in msg:
                content = (msg.get("content") or "").strip()
                if content:
                    clean.append({
                        "role": "assistant",
                        "content": content,
                    })
            else:
                clean.append({"role": role, "content": msg.get("content", "")})
        return clean

    @staticmethod
    def _is_repeating(messages: list[dict[str, Any]], window: int = 4) -> bool:
        """Check if the last N tool calls are identical (name + args)."""
        tool_msgs = [
            m for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        if len(tool_msgs) < 2:
            return False

        recent = tool_msgs[-window:]
        if len(recent) < 2:
            return False

        signatures = []
        for m in recent:
            calls = m.get("tool_calls", [])
            sig = tuple(
                (c.get("function", {}).get("name"), c.get("function", {}).get("arguments"))
                for c in calls
            )
            signatures.append(sig)

        return len(set(signatures)) == 1

    @staticmethod
    def _has_tool_call(messages: list[dict[str, Any]], tool_name: str) -> bool:
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls", []):
                if call.get("function", {}).get("name") == tool_name:
                    return True
        return False

    @classmethod
    def _should_retry_for_missed_send_email(cls, messages: list[dict[str, Any]], assistant_text: str) -> bool:
        if not assistant_text:
            return False

        text = assistant_text.lower()
        if "email" not in text:
            return False

        send_markers = (
            "send the email",
            "sending the email",
            "send this email",
            "send it to",
            "i will now send",
            "i'll now send",
            "using this account",
        )
        if not any(marker in text for marker in send_markers):
            return False

        return not cls._has_tool_call(messages, "send_email")
