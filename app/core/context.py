import sys
from typing import Optional, Any
from services.llm.client import EmbeddingClient, LlmClient, RerankClient, TranscriptionClient
from .config import settings

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

# Service Registry for dynamic plugin loading
_services: dict[str, Any] = {}


def register_service(id: str, service: Any):
    """Register a dynamic service instance (used by plugins)"""
    _services[id] = service


def get_service(id: str) -> Optional[Any]:
    """Get a registered service instance (used by plugins)"""
    return _services.get(id)


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
