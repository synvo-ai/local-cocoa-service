from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Literal

from core.config import settings
from core.context import get_indexer
from core.spawn_manager import get_spawn_manager, SpawnConfig

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
    # Sub-process settings
    active_model_id: Optional[str] = None
    active_embedding_model_id: Optional[str] = None
    active_reranker_model_id: Optional[str] = None
    active_audio_model_id: Optional[str] = None
    llm_context_tokens: Optional[int] = None


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
        # Sub-process settings
        "active_model_id": settings.active_model_id,
        "active_embedding_model_id": settings.active_embedding_model_id,
        "active_reranker_model_id": settings.active_reranker_model_id,
        "active_audio_model_id": settings.active_audio_model_id,
        "llm_context_tokens": settings.llm_context_tokens,
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

    # Handle model related changes
    manager = get_spawn_manager()
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

    # Restart spawns if needed
    for alias in restarts:
        if alias == 'vlm':
            descriptor = manager.get_descriptor(settings.active_model_id)
            mmproj_path = None
            if descriptor and (descriptor.get('type') == 'vlm' or descriptor.get('id') == 'vlm'):
                mmproj_id = descriptor.get('mmprojId') or 'vlm-mmproj'
                mmproj_path = manager.get_model_path(mmproj_id)
            await manager.stop_spawn('vlm')
            await manager.start_spawn(SpawnConfig(
                alias='vlm',
                model_path=manager.get_model_path(settings.active_model_id),
                port=settings.endpoints.vlm_port,
                context_size=settings.llm_context_tokens,
                threads=4,
                ngl=999,
                type='vlm',
                mmproj_path=mmproj_path
            ))
        elif alias == 'embedding':
            await manager.stop_spawn('embedding')
            await manager.start_spawn(SpawnConfig(
                alias='embedding',
                model_path=manager.get_model_path(settings.active_embedding_model_id),
                port=settings.endpoints.embedding_port,
                context_size=8192,
                threads=4,
                ngl=999,
                type='embedding',
                batch_size=8192,
                ubatch_size=512,
                parallel=4
            ))
        elif alias == 'reranker':
            await manager.stop_spawn('reranker')
            await manager.start_spawn(SpawnConfig(
                alias='reranker',
                model_path=manager.get_model_path(settings.active_reranker_model_id),
                port=settings.endpoints.reranker_port,
                context_size=4096,
                threads=2,
                ngl=999,
                type='reranking',
                ubatch_size=2048
            ))
        elif alias == 'whisper':
            await manager.stop_spawn('whisper')
            await manager.start_spawn(SpawnConfig(
                alias='whisper',
                model_path=manager.get_model_path(settings.active_audio_model_id),
                port=settings.endpoints.whisper_port,
                context_size=0,
                threads=4,
                ngl=0,
                type='whisper'
            ))

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
