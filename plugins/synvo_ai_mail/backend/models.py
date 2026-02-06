from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_serializer

class EmailAccountCreate(BaseModel):
    label: str
    protocol: Literal["imap", "pop3", "outlook"]
    host: Optional[str] = None
    port: int = Field(default=0, ge=0, le=65535)
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = True
    folder: Optional[str] = Field(default="INBOX")
    # Outlook specific
    client_id: Optional[str] = None
    tenant_id: Optional[str] = None


class EmailAccount(BaseModel):
    id: str
    label: str
    protocol: Literal["imap", "pop3", "outlook"]
    host: Optional[str] = None
    port: int
    username: Optional[str] = None
    secret: Optional[str] = None
    use_ssl: bool
    folder: Optional[str] = None
    enabled: bool = True
    created_at: dt.datetime
    updated_at: dt.datetime
    last_synced_at: Optional[dt.datetime] = None
    last_sync_status: Optional[str] = None
    # Outlook specific
    client_id: Optional[str] = None
    tenant_id: Optional[str] = None
    token_cache: Optional[str] = None


class EmailAccountSummary(BaseModel):
    id: str
    label: str
    protocol: Literal["imap", "pop3", "outlook"]
    host: str
    port: int
    username: str
    use_ssl: bool
    folder: Optional[str] = None
    enabled: bool
    created_at: dt.datetime
    updated_at: dt.datetime
    last_synced_at: Optional[dt.datetime]
    last_sync_status: Optional[str]
    total_messages: int
    recent_new_messages: int
    folder_id: str
    folder_path: Path

    @field_serializer("folder_path", when_used="json")
    def _serialize_folder_path(self, value: Path) -> str:
        return str(value)


class EmailSyncRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=500)


class EmailSyncResult(BaseModel):
    account_id: str
    folder_id: str
    folder_path: Path
    new_messages: int
    total_messages: int
    indexed: int
    last_synced_at: dt.datetime
    status: Literal["ok", "error"]
    message: Optional[str] = None

    @field_serializer("folder_path", when_used="json")
    def _serialize_folder_path(self, value: Path) -> str:
        return str(value)


class EmailMessageRecord(BaseModel):
    id: str
    account_id: str
    external_id: str
    subject: Optional[str]
    sender: Optional[str]
    recipients: list[str] = Field(default_factory=list)
    sent_at: Optional[dt.datetime]
    stored_path: Path
    size: int
    created_at: dt.datetime
    # Memory build status: 'pending', 'success', 'failed'
    memory_status: Optional[str] = None
    memory_error: Optional[str] = None
    memory_built_at: Optional[dt.datetime] = None

    @field_serializer("stored_path", when_used="json")
    def _serialize_stored_path(self, value: Path) -> str:
        return str(value)


class EmailMessageSummary(BaseModel):
    id: str
    account_id: str
    subject: Optional[str]
    sender: Optional[str]
    recipients: list[str] = Field(default_factory=list)
    sent_at: Optional[dt.datetime]
    stored_path: Path
    size: int
    created_at: dt.datetime
    preview: Optional[str] = None

    @field_serializer("stored_path", when_used="json")
    def _serialize_stored_path(self, value: Path) -> str:
        return str(value)


class EmailMessageContent(EmailMessageSummary):
    markdown: str
