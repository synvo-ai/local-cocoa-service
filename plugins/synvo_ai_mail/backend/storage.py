"""Email storage operations."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Optional

from services.storage import StorageBase
from .models import EmailAccount, EmailMessageRecord


class EmailMixin(StorageBase):
    """Mixin for handling email accounts and messages."""
    plugin_id: str = ""

    def __init__(self, plugin_id: str, db_path: str = "") -> None:
        # Initialize storage via inherited StorageBase
        super().__init__(db_path=db_path)
        self.plugin_id = plugin_id
        
        # Initialize database tables
        with self.connect() as conn:
            self._ensure_email_columns(conn)
    
    def _get_account_table_name(self) -> str:
        return f"{self.plugin_id}_email_accounts"
    
    def _get_message_table_name(self) -> str:
        return f"{self.plugin_id}_email_messages"

    def _ensure_email_columns(self, conn: sqlite3.Connection) -> None:
        account_table = self._get_account_table_name()
        message_table = self._get_message_table_name()
        
        # Ensure tables exist
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {account_table} (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                protocol TEXT NOT NULL,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                username TEXT NOT NULL,
                secret TEXT NOT NULL,
                use_ssl INTEGER NOT NULL,
                folder TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_synced_at TEXT,
                last_sync_status TEXT,
                client_id TEXT,
                tenant_id TEXT
            );
        """)

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {message_table} (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                external_id TEXT NOT NULL,
                subject TEXT,
                sender TEXT,
                recipients TEXT,
                sent_at TEXT,
                stored_path TEXT NOT NULL,
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                memory_status TEXT,
                memory_error TEXT,
                memory_built_at TEXT,
                FOREIGN KEY(account_id) REFERENCES {account_table}(id) ON DELETE CASCADE
            );
        """)
        
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{message_table}_account_id ON {message_table}(account_id);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{message_table}_external_id ON {message_table}(external_id);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{message_table}_created_at ON {message_table}(created_at);")

        accounts = {row["name"] for row in conn.execute(f"PRAGMA table_info({account_table})").fetchall()}
        messages = {row["name"] for row in conn.execute(f"PRAGMA table_info({message_table})").fetchall()}

        def add_account_column(name: str, definition: str) -> None:
            if name not in accounts:
                conn.execute(f"ALTER TABLE {account_table} ADD COLUMN {name} {definition}")

        def add_message_column(name: str, definition: str) -> None:
            if name not in messages:
                conn.execute(f"ALTER TABLE {message_table} ADD COLUMN {name} {definition}")

        add_account_column("last_synced_at", "TEXT")
        add_account_column("last_sync_status", "TEXT")
        add_account_column("enabled", "INTEGER NOT NULL DEFAULT 1")
        add_message_column("recipients", "TEXT")
        add_message_column("created_at", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")
        # Memory build status columns
        add_message_column("memory_status", "TEXT")  # 'pending', 'success', 'failed'
        add_message_column("memory_error", "TEXT")
        add_message_column("memory_built_at", "TEXT")
        # Migrations that were inline
        add_account_column("client_id", "TEXT")
        add_account_column("tenant_id", "TEXT")

    def list_email_accounts(self) -> list[EmailAccount]:
        account_table = self._get_account_table_name()
        with self.connect() as conn:  # type: ignore
            rows = conn.execute(
                f"SELECT * FROM {account_table} ORDER BY created_at ASC",
            ).fetchall()
        return [self._row_to_email_account(row) for row in rows]

    def get_email_account(self, account_id: str) -> Optional[EmailAccount]:
        account_table = self._get_account_table_name()
        with self.connect() as conn:  # type: ignore
            row = conn.execute(f"SELECT * FROM {account_table} WHERE id = ?", (account_id,)).fetchone()
        return self._row_to_email_account(row) if row else None

    def upsert_email_account(self, account: EmailAccount) -> None:
        account_table = self._get_account_table_name()
        with self.connect() as conn:  # type: ignore
            conn.execute(
                f"""
                INSERT INTO {account_table} (
                    id, label, protocol, host, port, username, secret, use_ssl, folder, enabled,
                    created_at, updated_at, last_synced_at, last_sync_status, client_id, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label=excluded.label,
                    protocol=excluded.protocol,
                    host=excluded.host,
                    port=excluded.port,
                    username=excluded.username,
                    secret=excluded.secret,
                    use_ssl=excluded.use_ssl,
                    folder=excluded.folder,
                    enabled=excluded.enabled,
                    updated_at=excluded.updated_at,
                    last_synced_at=excluded.last_synced_at,
                    last_sync_status=excluded.last_sync_status,
                    client_id=excluded.client_id,
                    tenant_id=excluded.tenant_id
                """,
                (
                    account.id,
                    account.label,
                    account.protocol,
                    account.host,
                    account.port,
                    account.username,
                    account.secret,
                    1 if account.use_ssl else 0,
                    account.folder,
                    1 if account.enabled else 0,
                    account.created_at.isoformat(),
                    account.updated_at.isoformat(),
                    account.last_synced_at.isoformat() if account.last_synced_at else None,
                    account.last_sync_status,
                    account.client_id,
                    account.tenant_id,
                ),
            )

    def delete_email_account(self, account_id: str) -> None:
        account_table = self._get_account_table_name()
        with self.connect() as conn:  # type: ignore
            conn.execute(f"DELETE FROM {account_table} WHERE id = ?", (account_id,))

    def update_email_account_sync(self, account_id: str, *, last_synced_at: Optional[dt.datetime], status: Optional[str]) -> None:
        account_table = self._get_account_table_name()
        with self.connect() as conn:  # type: ignore
            conn.execute(
                f"""
                UPDATE {account_table}
                SET last_synced_at = ?, last_sync_status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    last_synced_at.isoformat() if last_synced_at else None,
                    status,
                    dt.datetime.now(dt.timezone.utc).isoformat(),
                    account_id,
                ),
            )

    def list_email_message_ids(self, account_id: str) -> set[str]:
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            rows = conn.execute(
                f"SELECT external_id FROM {message_table} WHERE account_id = ?",
                (account_id,),
            ).fetchall()
        return {str(row["external_id"]) for row in rows}

    def record_email_message(self, record: EmailMessageRecord) -> None:
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            conn.execute(
                f"""
                INSERT OR IGNORE INTO {message_table} (
                    id, account_id, external_id, subject, sender, recipients, sent_at, stored_path, size, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.account_id,
                    record.external_id,
                    record.subject,
                    record.sender,
                    json.dumps(record.recipients, ensure_ascii=False),
                    record.sent_at.isoformat() if record.sent_at else None,
                    str(record.stored_path),
                    record.size,
                    record.created_at.isoformat(),
                ),
            )

    def list_email_messages(self, account_id: str, limit: int = 100) -> list[EmailMessageRecord]:
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            rows = conn.execute(
                f"""
                SELECT * FROM {message_table}
                WHERE account_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (account_id, limit),
            ).fetchall()
        return [self._row_to_email_message(row) for row in rows]

    def get_email_message(self, message_id: str) -> Optional[EmailMessageRecord]:
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            row = conn.execute(
                f"SELECT * FROM {message_table} WHERE id = ?",
                (message_id,),
            ).fetchone()
        return self._row_to_email_message(row) if row else None

    def count_email_messages(self, account_id: str) -> int:
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            row = conn.execute(
                f"SELECT COUNT(*) FROM {message_table} WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        return int(row[0] if row else 0)

    def count_email_messages_since(self, account_id: str, threshold: dt.datetime) -> int:
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            row = conn.execute(
                f"SELECT COUNT(*) FROM {message_table} WHERE account_id = ? AND created_at >= ?",
                (account_id, threshold.isoformat()),
            ).fetchone()
        return int(row[0] if row else 0)

    def prune_missing_email_messages(self, account_id: Optional[str] = None) -> int:
        message_table = self._get_message_table_name()
        removed = 0
        query = f"SELECT id, stored_path FROM {message_table}"
        params: tuple[object, ...] = ()
        if account_id:
            query += " WHERE account_id = ?"
            params = (account_id,)

        with self.connect() as conn:  # type: ignore
            rows = conn.execute(query, params).fetchall()
            orphan_ids = [row["id"] for row in rows if not Path(row["stored_path"]).exists()]
            if orphan_ids:
                conn.executemany(f"DELETE FROM {message_table} WHERE id = ?", ((identifier,) for identifier in orphan_ids))
                removed = len(orphan_ids)
        return removed

    def update_email_memory_status(
        self, 
        message_id: str, 
        status: str, 
        error: Optional[str] = None
    ) -> None:
        """Update memory build status for an email message.
        
        Args:
            message_id: Email message ID
            status: 'pending', 'success', or 'failed'
            error: Error message if status is 'failed'
        """
        message_table = self._get_message_table_name()
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self.connect() as conn:  # type: ignore
            # self._ensure_email_columns(conn) # Removed recursive call to avoid overhead, assuming init happened
            conn.execute(
                f"""
                UPDATE {message_table} 
                SET memory_status = ?, memory_error = ?, memory_built_at = ?
                WHERE id = ?
                """,
                (status, error, now if status in ('success', 'failed') else None, message_id),
            )

    def list_failed_email_messages(self, account_id: str) -> list[EmailMessageRecord]:
        """Get email messages that failed memory build."""
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            rows = conn.execute(
                f"SELECT * FROM {message_table} WHERE account_id = ? AND memory_status = 'failed' ORDER BY created_at DESC",
                (account_id,),
            ).fetchall()
        return [self._row_to_email_message(row) for row in rows]

    def list_pending_email_messages(self, account_id: str) -> list[EmailMessageRecord]:
        """Get email messages that haven't been processed yet."""
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            rows = conn.execute(
                f"SELECT * FROM {message_table} WHERE account_id = ? AND (memory_status IS NULL OR memory_status = 'pending') ORDER BY created_at DESC",
                (account_id,),
            ).fetchall()
        return [self._row_to_email_message(row) for row in rows]

    def reset_email_memory_status(self, message_id: str) -> None:
        """Reset memory status to pending for retry."""
        message_table = self._get_message_table_name()
        with self.connect() as conn:  # type: ignore
            conn.execute(
                f"UPDATE {message_table} SET memory_status = 'pending', memory_error = NULL, memory_built_at = NULL WHERE id = ?",
                (message_id,),
            )

    @staticmethod
    def _row_to_email_account(row: sqlite3.Row) -> EmailAccount:
        return EmailAccount(
            id=row["id"],
            label=row["label"],
            protocol=row["protocol"],
            host=row["host"],
            port=int(row["port"]),
            username=row["username"],
            secret=row["secret"],
            use_ssl=bool(row["use_ssl"]),
            folder=row["folder"],
            enabled=bool(row["enabled"]),
            created_at=dt.datetime.fromisoformat(row["created_at"]),
            updated_at=dt.datetime.fromisoformat(row["updated_at"]),
            last_synced_at=dt.datetime.fromisoformat(row["last_synced_at"]) if row["last_synced_at"] else None,
            last_sync_status=row["last_sync_status"],
            client_id=row["client_id"] if "client_id" in row.keys() else None,
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else None,
        )

    @staticmethod
    def _row_to_email_message(row: sqlite3.Row) -> EmailMessageRecord:
        recipients_payload = row["recipients"]
        recipients: list[str] = []
        if recipients_payload:
            try:
                data = json.loads(recipients_payload)
                if isinstance(data, list):
                    recipients = [str(item) for item in data]
            except json.JSONDecodeError:
                recipients = [str(recipients_payload)]
        keys = row.keys()
        return EmailMessageRecord(
            id=row["id"],
            account_id=row["account_id"],
            external_id=row["external_id"],
            subject=row["subject"],
            sender=row["sender"],
            recipients=recipients,
            sent_at=dt.datetime.fromisoformat(row["sent_at"]) if row["sent_at"] else None,
            stored_path=Path(row["stored_path"]),
            size=int(row["size"]),
            created_at=dt.datetime.fromisoformat(row["created_at"]),
            memory_status=row["memory_status"] if "memory_status" in keys else None,
            memory_error=row["memory_error"] if "memory_error" in keys else None,
            memory_built_at=dt.datetime.fromisoformat(row["memory_built_at"]) if "memory_built_at" in keys and row["memory_built_at"] else None,
        )
