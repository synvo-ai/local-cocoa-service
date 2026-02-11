from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_serializer

class NoteCreate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None


class NoteRecord(BaseModel):
    id: str
    title: str
    path: Path
    created_at: dt.datetime
    updated_at: dt.datetime

    @field_serializer("path", when_used="json")
    def _serialize_path(self, value: Path) -> str:
        return str(value)


class NoteSummary(BaseModel):
    id: str
    title: str
    updated_at: dt.datetime
    preview: Optional[str] = None


class NoteContent(BaseModel):
    id: str
    title: str
    markdown: str
    created_at: dt.datetime
    updated_at: dt.datetime
