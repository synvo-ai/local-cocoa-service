from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from core.config import settings
from core.context import get_search_engine
from core.models import QaRequest

logger = logging.getLogger(__name__)


@dataclass
class TelegramState:
    enabled: bool = False
    running: bool = False
    paired_chat_ids: set[int] = field(default_factory=set)
    last_update_id: int = 0
    poll_interval_seconds: int = 3
    backend_url: str = "http://127.0.0.1:8890/plugins/telegram"


class TelegramService:
    def __init__(self) -> None:
        self._token_file = settings.paths.runtime_root / "telegram" / "token.txt"
        token = self._load_token()
        self._token = token
        self._base_url = f"https://api.telegram.org/bot{token}" if token else ""
        self._state = TelegramState(enabled=bool(token), running=False)
        self._messages: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._poll_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._state_file = settings.paths.runtime_root / "telegram" / "state.json"
        self._load_state()
        self._search_engine = get_search_engine()

    def _load_token(self) -> str:
        runtime_token = ""
        try:
            if self._token_file.exists():
                runtime_token = self._token_file.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Failed to load runtime Telegram token: %s", exc)
        env_token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        return runtime_token or env_token

    def _save_token(self, token: str) -> None:
        self._token_file.parent.mkdir(parents=True, exist_ok=True)
        self._token_file.write_text(token, encoding="utf-8")

    def _clear_token(self) -> None:
        try:
            if self._token_file.exists():
                self._token_file.unlink()
        except Exception as exc:
            logger.warning("Failed to clear runtime Telegram token: %s", exc)

    @staticmethod
    def _mask_token(token: str) -> str:
        if not token:
            return ""
        if len(token) <= 8:
            return "*" * len(token)
        return f"{token[:4]}...{token[-4:]}"

    def _load_state(self) -> None:
        try:
            if not self._state_file.exists():
                return
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            paired = raw.get("paired_chat_ids", [])
            self._state.paired_chat_ids = {int(x) for x in paired if int(x) > 0}
            self._state.last_update_id = max(0, int(raw.get("last_update_id", 0)))
        except Exception as exc:
            logger.warning("Failed to load telegram state: %s", exc)

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps(
                    {
                        "paired_chat_ids": sorted(self._state.paired_chat_ids),
                        "last_update_id": self._state.last_update_id,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save telegram state: %s", exc)

    async def start(self) -> None:
        async with self._lock:
            if not self._state.enabled:
                self._state.running = False
                logger.info("Telegram service disabled: TELEGRAM_BOT_TOKEN is not set")
                return
            if self._poll_task and not self._poll_task.done():
                self._state.running = True
                return
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
            self._poll_task = asyncio.create_task(self._poll_loop(), name="telegram-poll-loop")
            self._state.running = True
            logger.info("Telegram poller started")

    async def stop(self) -> None:
        task: asyncio.Task[None] | None = None
        client: httpx.AsyncClient | None = None
        async with self._lock:
            task = self._poll_task
            self._poll_task = None
            client = self._client
            self._client = None
            self._state.running = False
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("Telegram poller stop error: %s", exc)
        if client:
            await client.aclose()
        self._save_state()
        logger.info("Telegram poller stopped")

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    async def get_status(self) -> dict[str, Any]:
        async with self._lock:
            has_runtime_token = self._token_file.exists()
            return {
                "enabled": self._state.enabled,
                "running": self._state.running,
                "paired_chats": sorted(self._state.paired_chat_ids),
                "last_update_id": self._state.last_update_id,
                "poll_interval_seconds": self._state.poll_interval_seconds,
                "backend_url": self._state.backend_url,
                "token_preview": self._mask_token(self._token),
                "token_source": "runtime" if has_runtime_token else ("env" if self._token else "none"),
            }

    async def set_token(self, token: str) -> dict[str, Any]:
        cleaned = (token or "").strip()
        if not cleaned:
            await self.stop()
            async with self._lock:
                self._token = ""
                self._base_url = ""
                self._state.enabled = False
                self._state.running = False
            self._clear_token()
            return await self.get_status()

        # Validate token with Telegram before persisting.
        await self._validate_token(cleaned)
        async with self._lock:
            self._token = cleaned
            self._base_url = f"https://api.telegram.org/bot{cleaned}"
            self._state.enabled = True
        self._save_token(cleaned)
        await self.restart()
        return await self.get_status()

    async def _validate_token(self, token: str) -> None:
        base_url = f"https://api.telegram.org/bot{token}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=8.0)) as client:
            response = await client.get(f"{base_url}/getMe")
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                raise RuntimeError("Invalid Telegram bot token.")

    async def list_messages(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:
            safe_limit = max(1, min(int(limit), 500))
            return list(self._messages[-safe_limit:])

    async def pair_chat(self, chat_id: int) -> TelegramState:
        async with self._lock:
            self._state.paired_chat_ids = {chat_id}
            self._save_state()
            return self._state

    async def unpair_chat(self, chat_id: int) -> TelegramState:
        async with self._lock:
            self._state.paired_chat_ids.discard(chat_id)
            self._save_state()
            return self._state

    async def send_test_message(self, chat_id: int | None, text: str | None) -> None:
        async with self._lock:
            if chat_id is None:
                chat_id = next(iter(self._state.paired_chat_ids), None)
                if chat_id is None:
                    raise RuntimeError("No paired chat available. Send /pair to the bot first.")
            if chat_id not in self._state.paired_chat_ids:
                raise RuntimeError("Chat is not paired.")

        await self._send_message(chat_id, text or "Test message from Local Cocoa.")

    async def _poll_loop(self) -> None:
        try:
            while True:
                await self._poll_once()
                await asyncio.sleep(self._state.poll_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Telegram poll loop crashed: %s", exc)
        finally:
            async with self._lock:
                self._state.running = False

    async def _poll_once(self) -> None:
        async with self._lock:
            if not self._client:
                return
            offset = self._state.last_update_id + 1
            client = self._client

        response = await client.get(
            f"{self._base_url}/getUpdates",
            params={"offset": offset, "timeout": 20, "allowed_updates": json.dumps(["message"])},
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram getUpdates failed: {payload}")

        updates = payload.get("result", [])
        if not updates:
            return

        for update in updates:
            update_id = int(update.get("update_id", 0))
            message = update.get("message") or {}
            await self._handle_incoming_message(message)
            async with self._lock:
                if update_id > self._state.last_update_id:
                    self._state.last_update_id = update_id
        self._save_state()

    async def _handle_incoming_message(self, message: dict[str, Any]) -> None:
        chat = message.get("chat") or {}
        chat_id = int(chat.get("id", 0) or 0)
        if chat_id <= 0:
            return

        text = str(message.get("text") or "").strip()
        msg = {
            "chat_id": chat_id,
            "message_id": message.get("message_id"),
            "date": message.get("date"),
            "text": text,
            "from": message.get("from"),
            "chat": chat,
            "role": "user",
            "streaming": False,
        }
        await self._append_message(msg)

        lowered = text.lower()
        if lowered.startswith("/start"):
            await self._send_message(
                chat_id,
                "Local Cocoa bot is online. Send /pair to connect this chat.",
            )
            return

        if lowered.startswith("/pair"):
            async with self._lock:
                self._state.paired_chat_ids = {chat_id}
                self._save_state()
            await self._send_message(chat_id, "Paired with Local Cocoa. You can now send requests.")
            return

        if lowered.startswith("/unpair"):
            async with self._lock:
                self._state.paired_chat_ids.discard(chat_id)
                self._save_state()
            await self._send_message(chat_id, "Unpaired from Local Cocoa.")
            return

        async with self._lock:
            is_paired = chat_id in self._state.paired_chat_ids
        if is_paired:
            await self._answer_user_message(chat_id, text)

    async def _answer_user_message(self, chat_id: int, text: str) -> None:
        query = (text or "").strip()
        if not query:
            return
        stream_message_id = -int(time.time() * 1000)
        await self._append_message(
            {
                "chat_id": chat_id,
                "message_id": stream_message_id,
                "date": int(time.time()),
                "text": "",
                "from": {"first_name": "Local Cocoa", "username": "local_cocoa_bot"},
                "chat": {"title": "Local Cocoa"},
                "role": "assistant",
                "streaming": True,
            }
        )
        answer = ""
        try:
            async for event_str in self._search_engine.stream_answer(
                QaRequest(
                    query=query,
                    mode="qa",
                    search_mode="auto",
                    use_vision_for_answer=False,
                )
            ):
                try:
                    event = json.loads(event_str)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "token":
                    chunk = str(event.get("data") or "")
                    if chunk:
                        answer += chunk
                        await self._update_stream_message(
                            stream_message_id, answer, streaming=True
                        )
                elif etype == "error":
                    err = str(event.get("data") or "Unknown error")
                    answer += f"\nError: {err}"
                    await self._update_stream_message(
                        stream_message_id, answer, streaming=True
                    )
                elif etype == "done":
                    break
        except Exception as exc:
            logger.exception("Telegram QA failed: %s", exc)
            answer = f"Failed to answer right now: {str(exc)}"

        answer = answer.strip() or "I could not find a clear answer."

        # Telegram text messages are limited to 4096 chars.
        if len(answer) > 3900:
            answer = answer[:3900] + "\n\n[truncated]"
        await self._update_stream_message(stream_message_id, answer, streaming=False)
        try:
            await self._send_message(chat_id, answer, track_history=False)
        except Exception as exc:
            logger.warning("Failed to send Telegram answer: %s", exc)
            await self._update_stream_message(
                stream_message_id,
                f"{answer}\n\n[Telegram send failed: {exc}]",
                streaming=False,
            )

    async def _send_message(self, chat_id: int, text: str, track_history: bool = True) -> None:
        async with self._lock:
            client = self._client
            enabled = self._state.enabled
        if not enabled or not client:
            raise RuntimeError("Telegram service is not running. Check bot token and poller status.")

        response = await client.post(
            f"{self._base_url}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram sendMessage failed: {payload}")
        if track_history:
            result = payload.get("result") or {}
            await self._append_message(
                {
                    "chat_id": chat_id,
                    "message_id": result.get("message_id", int(time.time() * 1000)),
                    "date": result.get("date", int(time.time())),
                    "text": str(result.get("text") or text),
                    "from": result.get("from")
                    or {"first_name": "Local Cocoa", "username": "local_cocoa_bot"},
                    "chat": result.get("chat") or {"title": "Local Cocoa"},
                    "role": "assistant",
                    "streaming": False,
                }
            )

    async def _append_message(self, msg: dict[str, Any]) -> None:
        async with self._lock:
            self._messages.append(msg)
            if len(self._messages) > 500:
                self._messages = self._messages[-500:]

    async def _update_stream_message(
        self, message_id: int, text: str, streaming: bool
    ) -> None:
        async with self._lock:
            for msg in reversed(self._messages):
                if msg.get("message_id") == message_id:
                    msg["text"] = text
                    msg["streaming"] = streaming
                    msg["role"] = "assistant"
                    break


telegram_service = TelegramService()
