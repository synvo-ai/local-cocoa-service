"""Note storage operations."""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Optional

from services.storage import StorageBase
from .models import NoteRecord


class NoteMixin(StorageBase):
    """Mixin for handling notes."""
    plugin_id: str = ""
    def __init__(self, plugin_id: str, db_path: str = "") -> None:
        # Initialize storage via inherited StorageBase
        super().__init__(db_path=db_path)
        self.plugin_id = plugin_id
        
        # Initialize database tables
        with self.connect() as conn:
            self._ensure_note_columns(conn)
    
    def _get_table_name(self) -> str:
        return f"{self.plugin_id}_notes"

    def _ensure_note_columns(self, conn: sqlite3.Connection) -> None:
        table_name = self._get_table_name()
        
        # Ensure table exists
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_updated ON {table_name}(updated_at DESC)")
        
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}

        def add_column(name: str, definition: str) -> None:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {definition}")

        add_column("title", "TEXT NOT NULL")
        add_column("path", "TEXT NOT NULL")
        add_column("created_at", "TEXT NOT NULL")
        add_column("updated_at", "TEXT NOT NULL")

    def db_upsert_note(self, record: NoteRecord) -> None:
        table_name = self._get_table_name()
        with self.connect() as conn:  # type: ignore
            conn.execute(f"""
                INSERT INTO {table_name} (id, title, path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    path=excluded.path,
                    updated_at=excluded.updated_at
                """,
                (
                    record.id,
                    record.title,
                    str(record.path),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )

    def db_list_notes(self) -> list[NoteRecord]:
        table_name = self._get_table_name()
        with self.connect() as conn:  # type: ignore
            rows = conn.execute(
                f"SELECT * FROM {table_name} ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_note(row) for row in rows]

    def db_get_note(self, note_id: str) -> Optional[NoteRecord]:
        table_name = self._get_table_name()
        with self.connect() as conn:  # type: ignore
            row = conn.execute(f"SELECT * FROM {table_name} WHERE id = ?", (note_id,)).fetchone()
        return self._row_to_note(row) if row else None

    def db_delete_note(self, note_id: str) -> None:
        table_name = self._get_table_name()
        with self.connect() as conn:  # type: ignore
            conn.execute(f"DELETE FROM {table_name} WHERE id = ?", (note_id,))

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> NoteRecord:
        return NoteRecord(
            id=row["id"],
            title=row["title"],
            path=Path(row["path"]),
            created_at=dt.datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')),
            updated_at=dt.datetime.fromisoformat(row["updated_at"].replace('Z', '+00:00')),
        )
