"""
Gemini Native Function Calling Client

Uses google-genai SDK for native tool calling (not OpenAI-compat).
This gives reliable structured tool_calls instead of text-based <tool_call> parsing.

Usage:
    client = GeminiNativeClient(api_key="...", model="gemini-2.5-flash")
    tools = [ToolSpec(...), ...]  # from services.agent.models
    calls, text = await client.call_with_tools(messages, tools)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiNativeClient:
    """Thin wrapper around google-genai SDK for native function calling."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def _convert_tools(self, tool_specs: list) -> list[types.Tool]:
        """Convert ToolSpec list to Gemini function declarations."""
        declarations = []
        for spec in tool_specs:
            properties = {}
            required = []
            for p in spec.parameters:
                prop = {"type": p.type.upper(), "description": p.description}
                if p.enum:
                    prop["enum"] = p.enum
                properties[p.name] = types.Schema(**prop)
                if p.required:
                    required.append(p.name)

            decl = types.FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters=types.Schema(
                    type="OBJECT",
                    properties=properties,
                    required=required if required else None,
                ),
            )
            declarations.append(decl)
        return [types.Tool(function_declarations=declarations)]

    def _messages_to_contents(self, messages: list[dict[str, Any]]) -> tuple[str | None, list]:
        """Convert OpenAI-style messages to Gemini contents format."""
        system = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""

            if role == "system":
                system = content

            elif role == "assistant":
                # Check if this assistant message contains tool_calls
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    # Convert to Gemini function_call parts
                    parts = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        parts.append(types.Part.from_function_call(
                            name=name, args=args,
                        ))
                    if content:
                        parts.insert(0, types.Part.from_text(text=content))
                    contents.append(types.Content(role="model", parts=parts))
                elif content:
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)],
                    ))

            elif role == "tool":
                # Tool result → Gemini function_response
                tool_name = msg.get("name", "")
                tool_call_id = msg.get("tool_call_id", "")
                # Try to find the tool name from the call_id if name is missing
                if not tool_name and tool_call_id:
                    for prev in messages:
                        for tc in prev.get("tool_calls", []):
                            if tc.get("id") == tool_call_id:
                                tool_name = tc.get("function", {}).get("name", "tool")
                                break
                if not tool_name:
                    tool_name = "tool"
                # Parse content as JSON if possible for structured response
                try:
                    parsed = json.loads(content)
                    # FunctionResponse requires a dict, wrap lists
                    result_data = parsed if isinstance(parsed, dict) else {"result": parsed}
                except (json.JSONDecodeError, TypeError):
                    result_data = {"result": content[:4000] if len(content) > 4000 else content}
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=tool_name,
                        response=result_data,
                    )],
                ))

            else:
                if content:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)],
                    ))
        return system, contents

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tool_specs: list,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> tuple[list[dict], str]:
        """
        Call Gemini with native function calling.

        Returns:
            (tool_calls, assistant_text)
            tool_calls: list of {"id": str, "tool_name": str, "arguments": dict}
            assistant_text: any text content from the response
        """
        tools = self._convert_tools(tool_specs)
        system, contents = self._messages_to_contents(messages)

        config = types.GenerateContentConfig(
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system:
            config.system_instruction = system

        logger.info(f"Gemini native call: {self.model}, {len(tool_specs)} tools, {len(contents)} messages")

        # Run sync SDK call in thread to not block event loop
        def _call():
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        response = await asyncio.to_thread(_call)

        # Parse response
        tool_calls = []
        text_parts = []

        if not response.candidates:
            logger.warning("Gemini returned no candidates")
            return [], ""

        for part in response.candidates[0].content.parts:
            if part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                tool_calls.append({
                    "id": uuid.uuid4().hex[:8],
                    "tool_name": fc.name,
                    "arguments": args,
                })
                logger.info(f"Gemini native tool_call: {fc.name}({list(args.keys())})")
            elif part.text:
                text_parts.append(part.text)

        assistant_text = "\n".join(text_parts)

        if tool_calls:
            logger.info(f"Gemini returned {len(tool_calls)} native tool calls")
        else:
            logger.info(f"Gemini returned text only ({len(assistant_text)} chars)")

        return tool_calls, assistant_text

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        """Simple completion without tools."""
        system, contents = self._messages_to_contents(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system:
            config.system_instruction = system

        def _call():
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        response = await asyncio.to_thread(_call)

        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text or ""
        return ""
