from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Literal

from core.config import settings
from core.context import get_indexer
from core.model_manager import get_model_manager, ModelType

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    vision_max_pixels: Optional[int] = None
    video_max_pixels: Optional[int] = None
    embed_batch_size: Optional[int] = None
    embed_batch_delay_ms: Optional[int] = None
    vision_batch_delay_ms: Optional[int] = None
    search_result_limit: Optional[int] = None
    qa_context_limit: Optional[int] = None
    max_snippet_length: Optional[int] = None
    summary_max_tokens: Optional[int] = None
    pdf_one_chunk_per_page: Optional[bool] = None
    rag_chunk_size: Optional[int] = None
    rag_chunk_overlap: Optional[int] = None
    default_indexing_mode: Optional[Literal["fast", "deep"]] = None
    # Memory extraction settings
    enable_memory_extraction: Optional[bool] = None
    memory_extraction_stage: Optional[Literal["fast", "deep", "none"]] = None
    memory_chunk_size: Optional[int] = None  # 0=use original chunks, >0=custom size


@router.get("/")
async def get_settings():
    return {
        "vision_max_pixels": settings.vision_max_pixels,
        "video_max_pixels": settings.video_max_pixels,
        "embed_batch_size": settings.embed_batch_size,
        "embed_batch_delay_ms": settings.embed_batch_delay_ms,
        "vision_batch_delay_ms": settings.vision_batch_delay_ms,
        "search_result_limit": settings.search_result_limit,
        "qa_context_limit": settings.qa_context_limit,
        "max_snippet_length": settings.max_snippet_length,
        "summary_max_tokens": settings.summary_max_tokens,
        "pdf_one_chunk_per_page": settings.pdf_one_chunk_per_page,
        "rag_chunk_size": settings.rag_chunk_size,
        "rag_chunk_overlap": settings.rag_chunk_overlap,
        "default_indexing_mode": settings.default_indexing_mode,
        # Memory settings
        "enable_memory_extraction": settings.enable_memory_extraction,
        "memory_extraction_stage": settings.memory_extraction_stage,
        "memory_chunk_size": settings.memory_chunk_size,
    }


@router.patch("/")
async def update_settings(update: SettingsUpdate):
    if update.vision_max_pixels is not None:
        settings.vision_max_pixels = update.vision_max_pixels
    if update.video_max_pixels is not None:
        settings.video_max_pixels = update.video_max_pixels
    if update.embed_batch_size is not None:
        settings.embed_batch_size = update.embed_batch_size
    if update.embed_batch_delay_ms is not None:
        settings.embed_batch_delay_ms = update.embed_batch_delay_ms
    if update.vision_batch_delay_ms is not None:
        settings.vision_batch_delay_ms = update.vision_batch_delay_ms
    if update.search_result_limit is not None:
        settings.search_result_limit = update.search_result_limit
    if update.qa_context_limit is not None:
        settings.qa_context_limit = update.qa_context_limit
    if update.max_snippet_length is not None:
        settings.max_snippet_length = update.max_snippet_length
    if update.summary_max_tokens is not None:
        settings.summary_max_tokens = update.summary_max_tokens
    if update.pdf_one_chunk_per_page is not None:
        settings.pdf_one_chunk_per_page = update.pdf_one_chunk_per_page
    if update.rag_chunk_size is not None:
        settings.rag_chunk_size = update.rag_chunk_size
    if update.rag_chunk_overlap is not None:
        settings.rag_chunk_overlap = update.rag_chunk_overlap
    if update.default_indexing_mode is not None:
        settings.default_indexing_mode = update.default_indexing_mode
    if update.enable_memory_extraction is not None:
        settings.enable_memory_extraction = update.enable_memory_extraction
    if update.memory_extraction_stage is not None:
        settings.memory_extraction_stage = update.memory_extraction_stage
    if update.memory_chunk_size is not None:
        settings.memory_chunk_size = update.memory_chunk_size


    # Handle model related changes
    manager = get_model_manager()
    restarts = []

    if update.active_model_id is not None and update.active_model_id != settings.active_model_id:
        settings.active_model_id = update.active_model_id
        restarts.append('vlm')
    if update.llm_context_tokens is not None and update.llm_context_tokens != settings.llm_context_tokens:
        settings.llm_context_tokens = update.llm_context_tokens
        if 'vlm' not in restarts: restarts.append('vlm')

    if update.active_embedding_model_id is not None and update.active_embedding_model_id != settings.active_embedding_model_id:
        settings.active_embedding_model_id = update.active_embedding_model_id
        restarts.append('embedding')

    if update.active_reranker_model_id is not None and update.active_reranker_model_id != settings.active_reranker_model_id:
        settings.active_reranker_model_id = update.active_reranker_model_id
        restarts.append('reranker')

    if update.active_audio_model_id is not None and update.active_audio_model_id != settings.active_audio_model_id:
        settings.active_audio_model_id = update.active_audio_model_id
        restarts.append('whisper')

    settings.save_to_file()

    # Restart models if needed
    for alias in restarts:
        mtype = None
        if alias == 'vlm': mtype = ModelType.VISION
        elif alias == 'embedding': mtype = ModelType.EMBEDDING
        elif alias == 'reranker': mtype = ModelType.RERANK
        elif alias == 'whisper': mtype = ModelType.WHISPER
        
        if mtype:
            manager.stop_model(mtype)
            await manager.ensure_model(mtype)

    return {
        "status": "ok",
        "settings": {
            "vision_max_pixels": settings.vision_max_pixels,
            "video_max_pixels": settings.video_max_pixels,
            "embed_batch_size": settings.embed_batch_size,
            "embed_batch_delay_ms": settings.embed_batch_delay_ms,
            "vision_batch_delay_ms": settings.vision_batch_delay_ms,
            "search_result_limit": settings.search_result_limit,
            "qa_context_limit": settings.qa_context_limit,
            "max_snippet_length": settings.max_snippet_length,
            "summary_max_tokens": settings.summary_max_tokens,
            "pdf_one_chunk_per_page": settings.pdf_one_chunk_per_page,
            "rag_chunk_size": settings.rag_chunk_size,
            "rag_chunk_overlap": settings.rag_chunk_overlap,
            "default_indexing_mode": settings.default_indexing_mode,
            "active_model_id": settings.active_model_id,
            "active_embedding_model_id": settings.active_embedding_model_id,
            "active_reranker_model_id": settings.active_reranker_model_id,
            "active_audio_model_id": settings.active_audio_model_id,
            "llm_context_tokens": settings.llm_context_tokens,
        }
    }
