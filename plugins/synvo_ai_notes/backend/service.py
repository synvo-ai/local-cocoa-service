from __future__ import annotations

import datetime as dt
import logging
import uuid
from pathlib import Path

from core.config import settings
from plugins.plugin_config import load_plugin_config
from core.context import get_indexer, get_storage
from services.indexer import Indexer
from services.storage import IndexStorage
from .models import NoteContent, NoteCreate, NoteRecord, NoteSummary
from .storage import NoteMixin

logger = logging.getLogger(__name__)


class NotesServiceError(Exception):
    """Base error for notes service issues."""


class NoteNotFound(NotesServiceError):
    """Raised when a requested note cannot be located."""


class NotesService(NoteMixin):
    def __init__(self, indexer: Indexer, plugin_id: str = "") -> None:
        # Initialize storage via inherited StorageBase
        super().__init__(plugin_id, db_path=settings.db_path)
        
        self.indexer = indexer
        _config = load_plugin_config(__file__)
        self._root = _config.storage_root
        self._root.mkdir(parents=True, exist_ok=True)

    def list_notes(self) -> list[NoteSummary]:
        notes = self.db_list_notes()
        summaries: list[NoteSummary] = []
        for note in notes:
            preview = self._preview(note.path)
            summaries.append(
                NoteSummary(
                    id=note.id,
                    title=note.title,
                    updated_at=note.updated_at,
                    preview=preview,
                )
            )
        return summaries

    def create_note(self, payload: NoteCreate) -> NoteSummary:
        now = dt.datetime.now(dt.timezone.utc)
        title = self._normalise_title(payload.title)
        identifier = uuid.uuid4().hex
        filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{identifier}.md"
        path = self._root / filename
        markdown = payload.body if payload.body is not None else ""
        path.write_text(markdown, encoding="utf-8")
        record = NoteRecord(id=identifier, title=title, path=path, created_at=now, updated_at=now)
        self.db_upsert_note(record)
        preview = self._preview(path)
        return NoteSummary(id=record.id, title=record.title, updated_at=record.updated_at, preview=preview)

    def get_note(self, note_id: str) -> NoteContent:
        record = self.db_get_note(note_id)
        if not record:
            raise NoteNotFound("Note not found.")
        markdown = self._read(record.path)
        return NoteContent(
            id=record.id,
            title=record.title,
            markdown=markdown,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def update_note(self, note_id: str, payload: NoteCreate) -> NoteContent:
        record = self.db_get_note(note_id)
        if not record:
            raise NoteNotFound("Note not found.")
        title = self._normalise_title(payload.title) if payload.title is not None else record.title
        markdown = payload.body if payload.body is not None else self._read(record.path)
        record.path.write_text(markdown, encoding="utf-8")
        updated = NoteRecord(
            id=record.id,
            title=title,
            path=record.path,
            created_at=record.created_at,
            updated_at=dt.datetime.now(dt.timezone.utc),
        )
        self.db_upsert_note(updated)
        return NoteContent(
            id=updated.id,
            title=updated.title,
            markdown=markdown,
            created_at=updated.created_at,
            updated_at=updated.updated_at,
        )

    def delete_note(self, note_id: str) -> None:
        record = self.db_get_note(note_id)
        if not record:
            raise NoteNotFound("Note not found.")

        # Try to clean up vectors if this note was indexed as a file
        try:
            file_record = self.get_file_by_path(record.path)
            if file_record:
                self.indexer.vector_store.delete_by_filter(file_id=file_record.id)
                self.delete_file(file_record.id)
        except Exception as exc:
            logger.warning("Failed to cleanup vectors for note %s: %s", note_id, exc)

        try:
            record.path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to delete note file at %s", record.path)
        self.db_delete_note(note_id)

    @staticmethod
    def _normalise_title(title: str | None) -> str:
        fallback = "Untitled note"
        if not title:
            return fallback
        trimmed = title.strip()
        return trimmed or fallback

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise NoteNotFound("Note file missing.") from exc

    @staticmethod
    def _preview(path: Path, lines: int = 6) -> str | None:
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        snippet = [line for line in content.splitlines() if line.strip()][:lines]
        return "\n".join(snippet) if snippet else None

# Lazy initialization of service
notes_service_global: Optional[NotesService] = None

def get_notes_service() -> NotesService:
    global notes_service_global

    if notes_service_global is None:
        raise RuntimeError("Notes service not initialized")
    return notes_service_global


def init_plugin_service(indexer: Indexer, plugin_id: str = "") -> NotesService:
    global notes_service_global
    
    if notes_service_global is None:
        try:
            # NotesService now inherits from StorageBase and initializes its own connection
            notes_service_global = NotesService(indexer, plugin_id)
        except Exception as e:
            logger.warning(f"Failed to initialize global notes service: {e}")
            
    if notes_service_global:
        return notes_service_global
        
    raise RuntimeError("Notes service not initialized")
