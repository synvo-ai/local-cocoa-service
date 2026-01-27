from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from services.llm.client import EmbeddingClient, LlmClient, RerankClient, TranscriptionClient
from .config import settings

# Import plugin services - import from service module directly, NOT from __init__
# These are loaded lazily in get_xxx_service functions to avoid circular imports
# and to ensure the plugins directory is in sys.path first.

from services.indexer import Indexer
from services.search.engine import SearchEngine
from services.storage import IndexStorage
from .vector_store import get_vector_store

storage = IndexStorage(settings.db_path)
embedding_client = EmbeddingClient()
rerank_client = RerankClient()
llm_client = LlmClient()
transcription_client = TranscriptionClient() if settings.endpoints.transcribe_url else None
indexer = Indexer(
    storage,
    embedding_client=embedding_client,
    llm_client=llm_client,
    transcription_client=transcription_client,
)
search_engine = SearchEngine(storage, embedding_client, rerank_client, llm_client, vectors=get_vector_store())


def get_storage() -> IndexStorage:
    return storage


def get_indexer() -> Indexer:
    return indexer


def get_embedding_client() -> EmbeddingClient:
    return embedding_client


def get_rerank_client() -> RerankClient:
    return rerank_client


def get_llm_client() -> LlmClient:
    return llm_client


def get_search_engine() -> SearchEngine:
    return search_engine


def get_transcription_client() -> Optional[TranscriptionClient]:
    return transcription_client


_notes_service = None


def get_notes_service():
    """Get the singleton NotesService instance."""
    global _notes_service
    if _notes_service is None:
        from notes.backend.service import NotesService
        _notes_service = NotesService(storage, indexer)
    return _notes_service


_email_service = None


def get_email_service():
    """Get the singleton EmailService instance."""
    global _email_service
    if _email_service is None:
        from mail.backend.service import EmailService
        _email_service = EmailService(storage, indexer)
    return _email_service

