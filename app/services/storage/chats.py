"""Chat storage operations."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any, Optional

from core.models import ChatMessage, ChatMessageUpdate, ChatSession, SearchHit


class ChatMixin:
    """Mixin for handling chat sessions and messages."""

    def _ensure_chat_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()}

        def add_column(name: str, definition: str) -> None:
            if name not in existing:
                conn.execute(f"ALTER TABLE chat_messages ADD COLUMN {name} {definition}")

        add_column("is_multi_path", "INTEGER")
        add_column("thinking_steps", "TEXT")
        add_column("needs_user_decision", "INTEGER")
        add_column("resume_token", "TEXT")
        add_column("decision_message", "TEXT")
        add_column("tool_calls", "TEXT")

    def upsert_chat_session(self, session: ChatSession) -> None:
        with self.connect() as conn:  # type: ignore
            conn.execute(
                """
                INSERT INTO chat_sessions (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    updated_at=excluded.updated_at
                """,
                (
                    session.id,
                    session.title,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )

    def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        with self.connect() as conn:  # type: ignore
            row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return None

            messages_rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()

            messages = [self._row_to_chat_message(r) for r in messages_rows]

            return ChatSession(
                id=row["id"],
                title=row["title"],
                created_at=dt.datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')),
                updated_at=dt.datetime.fromisoformat(row["updated_at"].replace('Z', '+00:00')),
                messages=messages,
            )

    def list_chat_sessions(self, limit: int = 100, offset: int = 0) -> list[ChatSession]:
        with self.connect() as conn:  # type: ignore
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

            sessions = []
            for row in rows:
                messages_rows = conn.execute(
                    "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC",
                    (row["id"],),
                ).fetchall()
                messages = [self._row_to_chat_message(r) for r in messages_rows]

                sessions.append(ChatSession(
                    id=row["id"],
                    title=row["title"],
                    created_at=dt.datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')),
                    updated_at=dt.datetime.fromisoformat(row["updated_at"].replace('Z', '+00:00')),
                    messages=messages,
                ))
            return sessions

    def delete_chat_session(self, session_id: str) -> None:
        with self.connect() as conn:  # type: ignore
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

    def add_chat_message(self, message: ChatMessage) -> None:
        with self.connect() as conn:  # type: ignore
            conn.execute(
                """
                INSERT INTO chat_messages (
                    id, session_id, role, content, timestamp, meta, "references",
                    is_multi_path, thinking_steps, needs_user_decision, resume_token,
                    decision_message, tool_calls
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.role,
                    message.content,
                    message.timestamp.isoformat(),
                    message.meta,
                    self._serialize_references(message.references),
                    1 if message.is_multi_path else (0 if message.is_multi_path is False else None),
                    self._serialize_thinking_steps(message.thinking_steps),
                    1 if message.needs_user_decision else (0 if message.needs_user_decision is False else None),
                    message.resume_token,
                    message.decision_message,
                    self._serialize_json_list(message.tool_calls),
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (message.timestamp.isoformat(), message.session_id),
            )

    def update_chat_message(self, session_id: str, message_id: str, payload: ChatMessageUpdate) -> Optional[ChatMessage]:
        with self.connect() as conn:  # type: ignore
            row = conn.execute(
                "SELECT * FROM chat_messages WHERE id = ? AND session_id = ?",
                (message_id, session_id),
            ).fetchone()
            if not row:
                return None

            existing = self._row_to_chat_message(row)
            updated = ChatMessage(
                id=existing.id,
                session_id=existing.session_id,
                role=payload.role or existing.role,
                content=payload.content if payload.content is not None else existing.content,
                timestamp=payload.timestamp or existing.timestamp,
                meta=payload.meta if payload.meta is not None else existing.meta,
                references=payload.references if payload.references is not None else existing.references,
                is_multi_path=payload.is_multi_path if payload.is_multi_path is not None else existing.is_multi_path,
                thinking_steps=payload.thinking_steps if payload.thinking_steps is not None else existing.thinking_steps,
                needs_user_decision=(
                    payload.needs_user_decision
                    if payload.needs_user_decision is not None
                    else existing.needs_user_decision
                ),
                resume_token=payload.resume_token if payload.resume_token is not None else existing.resume_token,
                decision_message=(
                    payload.decision_message
                    if payload.decision_message is not None
                    else existing.decision_message
                ),
                tool_calls=payload.tool_calls if payload.tool_calls is not None else existing.tool_calls,
            )

            conn.execute(
                """
                UPDATE chat_messages
                SET role = ?, content = ?, timestamp = ?, meta = ?, "references" = ?,
                    is_multi_path = ?, thinking_steps = ?, needs_user_decision = ?,
                    resume_token = ?, decision_message = ?, tool_calls = ?
                WHERE id = ? AND session_id = ?
                """,
                (
                    updated.role,
                    updated.content,
                    updated.timestamp.isoformat(),
                    updated.meta,
                    self._serialize_references(updated.references),
                    1 if updated.is_multi_path else (0 if updated.is_multi_path is False else None),
                    self._serialize_thinking_steps(updated.thinking_steps),
                    1 if updated.needs_user_decision else (0 if updated.needs_user_decision is False else None),
                    updated.resume_token,
                    updated.decision_message,
                    self._serialize_json_list(updated.tool_calls),
                    message_id,
                    session_id,
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (dt.datetime.now(dt.timezone.utc).isoformat(), session_id),
            )
            return updated

    @staticmethod
    def _row_to_chat_message(row: sqlite3.Row) -> ChatMessage:
        is_multi_path = None
        if "is_multi_path" in row.keys() and row["is_multi_path"] is not None:
            is_multi_path = bool(row["is_multi_path"])

        thinking_steps = None
        if "thinking_steps" in row.keys() and row["thinking_steps"]:
            thinking_steps = ChatMixin._deserialize_thinking_steps(row["thinking_steps"])
        needs_user_decision = None
        if "needs_user_decision" in row.keys() and row["needs_user_decision"] is not None:
            needs_user_decision = bool(row["needs_user_decision"])
        tool_calls = None
        if "tool_calls" in row.keys() and row["tool_calls"]:
            tool_calls = ChatMixin._deserialize_json_list(row["tool_calls"])

        return ChatMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            timestamp=dt.datetime.fromisoformat(row["timestamp"].replace('Z', '+00:00')),
            meta=row["meta"],
            references=ChatMixin._deserialize_references(row["references"]),
            is_multi_path=is_multi_path,
            thinking_steps=thinking_steps,
            needs_user_decision=needs_user_decision,
            resume_token=row["resume_token"] if "resume_token" in row.keys() else None,
            decision_message=row["decision_message"] if "decision_message" in row.keys() else None,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _serialize_references(refs: Optional[list[SearchHit]]) -> Optional[str]:
        if not refs:
            return None
        return json.dumps([ref.dict() for ref in refs], ensure_ascii=False)

    @staticmethod
    def _deserialize_references(payload: Optional[str]) -> Optional[list[SearchHit]]:
        if not payload:
            return None
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                return [SearchHit(**item) for item in data]
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @staticmethod
    def _serialize_thinking_steps(steps: Optional[list[dict]]) -> Optional[str]:
        if not steps:
            return None
        return json.dumps(steps, ensure_ascii=False)

    @staticmethod
    def _deserialize_thinking_steps(payload: Optional[str]) -> Optional[list[dict]]:
        if not payload:
            return None
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @staticmethod
    def _serialize_json_list(items: Optional[list[dict[str, Any]]]) -> Optional[str]:
        if not items:
            return None
        return json.dumps(items, ensure_ascii=False)

    @staticmethod
    def _deserialize_json_list(payload: Optional[str]) -> Optional[list[dict[str, Any]]]:
        if not payload:
            return None
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
        return None
