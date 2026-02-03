"""Memory storage mixin for user memories (episodes, foresights, event_logs, profiles)."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MemCellRecord:
    """MemCell record - source data for memory extraction."""
    id: str
    user_id: str
    original_data: str  # JSON string
    summary: Optional[str] = None
    subject: Optional[str] = None
    file_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_ordinal: Optional[int] = None
    type: Optional[str] = None  # 'Document' or 'Conversation'
    keywords: Optional[List[str]] = None
    timestamp: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EpisodeRecord:
    """Episodic memory record."""
    id: str
    user_id: str
    summary: str
    episode: Optional[str] = None
    subject: Optional[str] = None
    timestamp: Optional[str] = None
    parent_memcell_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EventLogRecord:
    """Event log (atomic fact) record."""
    id: str
    user_id: str
    atomic_fact: str
    timestamp: Optional[str] = None
    parent_episode_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ForesightRecord:
    """Foresight (prospective memory) record."""
    id: str
    user_id: str
    content: str
    evidence: Optional[str] = None
    parent_episode_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProfileRecord:
    """User profile record."""
    user_id: str
    user_name: Optional[str] = None
    personality: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    hard_skills: Optional[List[Dict[str, str]]] = None
    soft_skills: Optional[List[Dict[str, str]]] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryMixin:
    """Mixin providing memory storage operations."""

    def _ensure_memory_schema(self, conn: sqlite3.Connection) -> None:
        """Create memory tables if they don't exist."""
        conn.executescript(
            """
            -- MemCells table (source data for memory extraction)
            CREATE TABLE IF NOT EXISTS memory_memcells (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                original_data TEXT NOT NULL,
                summary TEXT,
                subject TEXT,
                file_id TEXT,
                chunk_id TEXT,
                chunk_ordinal INTEGER,
                type TEXT,
                keywords TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_memcells_user ON memory_memcells(user_id);
            CREATE INDEX IF NOT EXISTS idx_memory_memcells_file ON memory_memcells(file_id);
            CREATE INDEX IF NOT EXISTS idx_memory_memcells_timestamp ON memory_memcells(timestamp DESC);

            -- Episodes table (episodic memories)
            CREATE TABLE IF NOT EXISTS memory_episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                episode TEXT,
                subject TEXT,
                timestamp TEXT NOT NULL,
                parent_memcell_id TEXT REFERENCES memory_memcells(id) ON DELETE SET NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_episodes_user ON memory_episodes(user_id);
            CREATE INDEX IF NOT EXISTS idx_memory_episodes_timestamp ON memory_episodes(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_episodes_memcell ON memory_episodes(parent_memcell_id);

            -- Event logs table (atomic facts)
            CREATE TABLE IF NOT EXISTS memory_event_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                atomic_fact TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parent_episode_id TEXT REFERENCES memory_episodes(id) ON DELETE SET NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_event_logs_user ON memory_event_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_memory_event_logs_episode ON memory_event_logs(parent_episode_id);
            CREATE INDEX IF NOT EXISTS idx_memory_event_logs_timestamp ON memory_event_logs(timestamp DESC);

            -- Foresights table (prospective memories)
            CREATE TABLE IF NOT EXISTS memory_foresights (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                evidence TEXT,
                parent_episode_id TEXT REFERENCES memory_episodes(id) ON DELETE SET NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_foresights_user ON memory_foresights(user_id);
            CREATE INDEX IF NOT EXISTS idx_memory_foresights_episode ON memory_foresights(parent_episode_id);

            -- User profiles table
            CREATE TABLE IF NOT EXISTS memory_profiles (
                user_id TEXT PRIMARY KEY,
                user_name TEXT,
                personality TEXT,
                interests TEXT,
                hard_skills TEXT,
                soft_skills TEXT,
                updated_at TEXT NOT NULL,
                metadata TEXT
            );

            -- FTS5 for memory search
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                content,
                user_id UNINDEXED,
                memory_type UNINDEXED,
                memory_id UNINDEXED,
                tokenize='porter unicode61'
            );
            """
        )

    # ==================== Episodes ====================

    def upsert_episode(self, record: EpisodeRecord) -> None:
        """Insert or update an episode."""
        now = datetime.now(timezone.utc).isoformat()
        timestamp = record.timestamp or now
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            self._migrate_memory_schema(conn)
            conn.execute(
                """
                INSERT INTO memory_episodes (id, user_id, summary, episode, subject, timestamp, parent_memcell_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary = excluded.summary,
                    episode = excluded.episode,
                    subject = excluded.subject,
                    timestamp = excluded.timestamp,
                    parent_memcell_id = excluded.parent_memcell_id,
                    metadata = excluded.metadata
                """,
                (record.id, record.user_id, record.summary, record.episode,
                 record.subject, timestamp, record.parent_memcell_id, metadata_json, now),
            )
            # Update FTS
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ? AND memory_type = 'episode'",
                (record.id,)
            )
            fts_content = f"{record.summary} {record.episode or ''} {record.subject or ''}"
            conn.execute(
                "INSERT INTO memory_fts (content, user_id, memory_type, memory_id) VALUES (?, ?, ?, ?)",
                (fts_content, record.user_id, "episode", record.id)
            )

    def get_episodes(self, user_id: str, limit: int = 50, offset: int = 0) -> List[EpisodeRecord]:
        """Get episodes for a user."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            self._migrate_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, summary, episode, subject, timestamp, parent_memcell_id, metadata
                FROM memory_episodes
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset),
            ).fetchall()

        return [
            EpisodeRecord(
                id=row["id"],
                user_id=row["user_id"],
                summary=row["summary"],
                episode=row["episode"],
                subject=row["subject"],
                timestamp=row["timestamp"],
                parent_memcell_id=row["parent_memcell_id"] if "parent_memcell_id" in row.keys() else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def count_episodes(self, user_id: str) -> int:
        """Count episodes for a user."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memory_episodes WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    def delete_episode(self, episode_id: str) -> None:
        """Delete an episode."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            conn.execute("DELETE FROM memory_episodes WHERE id = ?", (episode_id,))
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ? AND memory_type = 'episode'",
                (episode_id,)
            )

    # ==================== Event Logs ====================

    def upsert_event_log(self, record: EventLogRecord) -> None:
        """Insert or update an event log."""
        now = datetime.now(timezone.utc).isoformat()
        timestamp = record.timestamp or now
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            conn.execute(
                """
                INSERT INTO memory_event_logs (id, user_id, atomic_fact, timestamp, parent_episode_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    atomic_fact = excluded.atomic_fact,
                    timestamp = excluded.timestamp,
                    parent_episode_id = excluded.parent_episode_id,
                    metadata = excluded.metadata
                """,
                (record.id, record.user_id, record.atomic_fact, timestamp,
                 record.parent_episode_id, metadata_json, now),
            )
            # Update FTS
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ? AND memory_type = 'event_log'",
                (record.id,)
            )
            conn.execute(
                "INSERT INTO memory_fts (content, user_id, memory_type, memory_id) VALUES (?, ?, ?, ?)",
                (record.atomic_fact, record.user_id, "event_log", record.id)
            )

    def get_event_logs(self, user_id: str, limit: int = 100, offset: int = 0) -> List[EventLogRecord]:
        """Get event logs for a user."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, atomic_fact, timestamp, parent_episode_id, metadata
                FROM memory_event_logs
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset),
            ).fetchall()

        return [
            EventLogRecord(
                id=row["id"],
                user_id=row["user_id"],
                atomic_fact=row["atomic_fact"],
                timestamp=row["timestamp"],
                parent_episode_id=row["parent_episode_id"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def count_event_logs(self, user_id: str) -> int:
        """Count event logs for a user."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memory_event_logs WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    def delete_event_log(self, event_log_id: str) -> None:
        """Delete an event log."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            conn.execute("DELETE FROM memory_event_logs WHERE id = ?", (event_log_id,))
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ? AND memory_type = 'event_log'",
                (event_log_id,)
            )

    def delete_event_logs_by_episode(self, episode_id: str) -> None:
        """Delete all event logs linked to a specific episode."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            # Get IDs first for FTS cleanup
            rows = conn.execute(
                "SELECT id FROM memory_event_logs WHERE parent_episode_id = ?",
                (episode_id,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "DELETE FROM memory_fts WHERE memory_id = ? AND memory_type = 'event_log'",
                    (row["id"],)
                )
            # Delete the records
            conn.execute("DELETE FROM memory_event_logs WHERE parent_episode_id = ?", (episode_id,))

    # ==================== Foresights ====================

    def upsert_foresight(self, record: ForesightRecord) -> None:
        """Insert or update a foresight."""
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            conn.execute(
                """
                INSERT INTO memory_foresights (id, user_id, content, evidence, parent_episode_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    evidence = excluded.evidence,
                    parent_episode_id = excluded.parent_episode_id,
                    metadata = excluded.metadata
                """,
                (record.id, record.user_id, record.content, record.evidence,
                 record.parent_episode_id, metadata_json, now),
            )
            # Update FTS
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ? AND memory_type = 'foresight'",
                (record.id,)
            )
            fts_content = f"{record.content} {record.evidence or ''}"
            conn.execute(
                "INSERT INTO memory_fts (content, user_id, memory_type, memory_id) VALUES (?, ?, ?, ?)",
                (fts_content, record.user_id, "foresight", record.id)
            )

    def get_foresights(self, user_id: str, limit: int = 50) -> List[ForesightRecord]:
        """Get foresights for a user."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, content, evidence, parent_episode_id, metadata
                FROM memory_foresights
                WHERE user_id = ?
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        return [
            ForesightRecord(
                id=row["id"],
                user_id=row["user_id"],
                content=row["content"],
                evidence=row["evidence"],
                parent_episode_id=row["parent_episode_id"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def count_foresights(self, user_id: str) -> int:
        """Count foresights for a user."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memory_foresights WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    # ==================== Profiles ====================

    def upsert_profile(self, record: ProfileRecord) -> None:
        """Insert or update a user profile."""
        now = datetime.now(timezone.utc).isoformat()

        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            conn.execute(
                """
                INSERT INTO memory_profiles (user_id, user_name, personality, interests, hard_skills, soft_skills, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    user_name = excluded.user_name,
                    personality = excluded.personality,
                    interests = excluded.interests,
                    hard_skills = excluded.hard_skills,
                    soft_skills = excluded.soft_skills,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    record.user_id,
                    record.user_name,
                    json.dumps(record.personality) if record.personality else None,
                    json.dumps(record.interests) if record.interests else None,
                    json.dumps(record.hard_skills) if record.hard_skills else None,
                    json.dumps(record.soft_skills) if record.soft_skills else None,
                    now,
                    json.dumps(record.metadata) if record.metadata else None,
                ),
            )

    def get_profile(self, user_id: str) -> Optional[ProfileRecord]:
        """Get a user profile."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            row = conn.execute(
                "SELECT * FROM memory_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if not row:
            return None

        return ProfileRecord(
            user_id=row["user_id"],
            user_name=row["user_name"],
            personality=json.loads(row["personality"]) if row["personality"] else None,
            interests=json.loads(row["interests"]) if row["interests"] else None,
            hard_skills=json.loads(row["hard_skills"]) if row["hard_skills"] else None,
            soft_skills=json.loads(row["soft_skills"]) if row["soft_skills"] else None,
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )

    # ==================== Search ====================

    def search_memories(self, user_id: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search across all memory types."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT memory_id, memory_type, content,
                       bm25(memory_fts) as score
                FROM memory_fts
                WHERE memory_fts MATCH ? AND user_id = ?
                ORDER BY score
                LIMIT ?
                """,
                (query, user_id, limit),
            ).fetchall()

        return [
            {
                "memory_id": row["memory_id"],
                "memory_type": row["memory_type"],
                "content": row["content"],
                "score": row["score"],
            }
            for row in rows
        ]

    # ==================== Schema Migration ====================

    def _migrate_memory_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing tables if they don't exist."""
        # Check and add parent_memcell_id to memory_episodes
        cursor = conn.execute("PRAGMA table_info(memory_episodes)")
        columns = {row[1] for row in cursor.fetchall()}
        if "parent_memcell_id" not in columns:
            conn.execute("ALTER TABLE memory_episodes ADD COLUMN parent_memcell_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_episodes_memcell ON memory_episodes(parent_memcell_id)")
            logger.info("Added parent_memcell_id column to memory_episodes")

    # ==================== MemCells ====================

    def upsert_memcell(self, record: MemCellRecord) -> None:
        """Insert or update a MemCell."""
        now = datetime.now(timezone.utc).isoformat()
        timestamp = record.timestamp or now
        created_at = record.created_at or now
        keywords_json = json.dumps(record.keywords) if record.keywords else None
        metadata_json = json.dumps(record.metadata) if record.metadata else None

        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            conn.execute(
                """
                INSERT INTO memory_memcells (id, user_id, original_data, summary, subject, file_id, chunk_id, chunk_ordinal, type, keywords, timestamp, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    original_data = excluded.original_data,
                    summary = excluded.summary,
                    subject = excluded.subject,
                    file_id = excluded.file_id,
                    chunk_id = excluded.chunk_id,
                    chunk_ordinal = excluded.chunk_ordinal,
                    type = excluded.type,
                    keywords = excluded.keywords,
                    timestamp = excluded.timestamp,
                    metadata = excluded.metadata
                """,
                (record.id, record.user_id, record.original_data, record.summary,
                 record.subject, record.file_id, record.chunk_id, record.chunk_ordinal,
                 record.type, keywords_json, timestamp, created_at, metadata_json),
            )
            # Update FTS
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ? AND memory_type = 'memcell'",
                (record.id,)
            )
            fts_content = f"{record.summary or ''} {record.subject or ''}"
            conn.execute(
                "INSERT INTO memory_fts (content, user_id, memory_type, memory_id) VALUES (?, ?, ?, ?)",
                (fts_content, record.user_id, "memcell", record.id)
            )

    def get_memcells(self, user_id: str, limit: int = 50, offset: int = 0) -> List[MemCellRecord]:
        """Get MemCells for a user."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, original_data, summary, subject, file_id, chunk_id, chunk_ordinal, type, keywords, timestamp, created_at, metadata
                FROM memory_memcells
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset),
            ).fetchall()

        return [
            MemCellRecord(
                id=row["id"],
                user_id=row["user_id"],
                original_data=row["original_data"],
                summary=row["summary"],
                subject=row["subject"],
                file_id=row["file_id"],
                chunk_id=row["chunk_id"],
                chunk_ordinal=row["chunk_ordinal"],
                type=row["type"],
                keywords=json.loads(row["keywords"]) if row["keywords"] else None,
                timestamp=row["timestamp"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def get_memcell(self, memcell_id: str) -> Optional[MemCellRecord]:
        """Get a single MemCell by ID."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            row = conn.execute(
                """
                SELECT id, user_id, original_data, summary, subject, file_id, chunk_id, chunk_ordinal, type, keywords, timestamp, created_at, metadata
                FROM memory_memcells
                WHERE id = ?
                """,
                (memcell_id,),
            ).fetchone()

        if not row:
            return None

        return MemCellRecord(
            id=row["id"],
            user_id=row["user_id"],
            original_data=row["original_data"],
            summary=row["summary"],
            subject=row["subject"],
            file_id=row["file_id"],
            chunk_id=row["chunk_id"],
            chunk_ordinal=row["chunk_ordinal"],
            type=row["type"],
            keywords=json.loads(row["keywords"]) if row["keywords"] else None,
            timestamp=row["timestamp"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )

    def get_memcells_by_file(self, file_id: str, limit: int = 100) -> List[MemCellRecord]:
        """Get MemCells for a specific file."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, original_data, summary, subject, file_id, chunk_id, chunk_ordinal, type, keywords, timestamp, created_at, metadata
                FROM memory_memcells
                WHERE file_id = ?
                ORDER BY chunk_ordinal ASC, timestamp DESC
                LIMIT ?
                """,
                (file_id, limit),
            ).fetchall()

        return [
            MemCellRecord(
                id=row["id"],
                user_id=row["user_id"],
                original_data=row["original_data"],
                summary=row["summary"],
                subject=row["subject"],
                file_id=row["file_id"],
                chunk_id=row["chunk_id"],
                chunk_ordinal=row["chunk_ordinal"],
                type=row["type"],
                keywords=json.loads(row["keywords"]) if row["keywords"] else None,
                timestamp=row["timestamp"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def count_memcells(self, user_id: str) -> int:
        """Count MemCells for a user."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memory_memcells WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    def get_episodes_by_memcell(self, memcell_id: str) -> List[EpisodeRecord]:
        """Get episodes linked to a specific MemCell."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            self._migrate_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, summary, episode, subject, timestamp, parent_memcell_id, metadata
                FROM memory_episodes
                WHERE parent_memcell_id = ?
                ORDER BY timestamp DESC
                """,
                (memcell_id,),
            ).fetchall()

        return [
            EpisodeRecord(
                id=row["id"],
                user_id=row["user_id"],
                summary=row["summary"],
                episode=row["episode"],
                subject=row["subject"],
                timestamp=row["timestamp"],
                parent_memcell_id=row["parent_memcell_id"] if "parent_memcell_id" in row.keys() else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def delete_memcell(self, memcell_id: str) -> None:
        """Delete a MemCell."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            conn.execute("DELETE FROM memory_memcells WHERE id = ?", (memcell_id,))
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ? AND memory_type = 'memcell'",
                (memcell_id,)
            )

    def get_memcells_by_chunk_id(self, chunk_id: str) -> List[MemCellRecord]:
        """Get MemCells by chunk_id (used for email memory status lookup)."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, original_data, summary, subject, file_id, chunk_id, chunk_ordinal, type, keywords, timestamp, created_at, metadata
                FROM memory_memcells
                WHERE chunk_id = ?
                ORDER BY timestamp DESC
                """,
                (chunk_id,),
            ).fetchall()

        return [
            MemCellRecord(
                id=row["id"],
                user_id=row["user_id"],
                original_data=row["original_data"],
                summary=row["summary"],
                subject=row["subject"],
                file_id=row["file_id"],
                chunk_id=row["chunk_id"],
                chunk_ordinal=row["chunk_ordinal"],
                type=row["type"],
                keywords=json.loads(row["keywords"]) if row["keywords"] else None,
                timestamp=row["timestamp"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def get_memcells_by_group_id(self, group_id: str) -> List[MemCellRecord]:
        """Get MemCells by group_id prefix in chunk_id (e.g., 'email_account::{account_id}' matches chunk_id starting with 'email_account_{account_id}_')."""
        # Convert group_id format to chunk_id prefix format
        # group_id: "email_account::{account_id}" -> chunk_id prefix: "email_account_{account_id}_"
        chunk_prefix = group_id.replace("::", "_") + "_"
        
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, original_data, summary, subject, file_id, chunk_id, chunk_ordinal, type, keywords, timestamp, created_at, metadata
                FROM memory_memcells
                WHERE chunk_id LIKE ?
                ORDER BY timestamp DESC
                """,
                (chunk_prefix + "%",),
            ).fetchall()

        return [
            MemCellRecord(
                id=row["id"],
                user_id=row["user_id"],
                original_data=row["original_data"],
                summary=row["summary"],
                subject=row["subject"],
                file_id=row["file_id"],
                chunk_id=row["chunk_id"],
                chunk_ordinal=row["chunk_ordinal"],
                type=row["type"],
                keywords=json.loads(row["keywords"]) if row["keywords"] else None,
                timestamp=row["timestamp"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def get_event_logs_by_episode(self, episode_id: str) -> List[EventLogRecord]:
        """Get event logs linked to a specific episode."""
        with self.connect() as conn:
            self._ensure_memory_schema(conn)
            rows = conn.execute(
                """
                SELECT id, user_id, atomic_fact, timestamp, parent_episode_id, metadata
                FROM memory_event_logs
                WHERE parent_episode_id = ?
                ORDER BY timestamp DESC
                """,
                (episode_id,),
            ).fetchall()

        return [
            EventLogRecord(
                id=row["id"],
                user_id=row["user_id"],
                atomic_fact=row["atomic_fact"],
                timestamp=row["timestamp"],
                parent_episode_id=row["parent_episode_id"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]
