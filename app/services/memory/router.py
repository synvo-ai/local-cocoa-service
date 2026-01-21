"""
Memory Router - FastAPI endpoints for memory management
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
from pydantic import BaseModel

from services.memory.models import (
    MemorizeRequest,
    MemorizeResult,
    SearchMemoryRequest,
    SearchMemoryResult,
    UserMemorySummary,
    EpisodeRecord,
    EventLogRecord,
    ForesightRecord,
    MemCellRecord,
    MemCellDetail,
    MemoryTypeEnum,
    RetrieveMethodEnum,
    BasicProfileResponse,
    SkillRecord,
    RawSystemDataRecord,
    ProfileTopic,
    ProfileSubtopic,
)
from services.memory.service import get_memory_service, MemoryServiceError, MemoryNotFound

router = APIRouter(prefix="/memory", tags=["memory"])


class ExtractMemoryRequest(BaseModel):
    """Request to manually extract memory from a file"""
    file_id: str
    user_id: str = "default_user"
    force: bool = False  # If True, re-extract even if already processed
    chunk_size: int | None = None  # Custom chunk size (chars). If set, concat all text and re-chunk


class ExtractMemoryResponse(BaseModel):
    """Response from memory extraction"""
    success: bool
    file_id: str
    message: str
    memcells_created: int = 0
    episodes_created: int = 0
    event_logs_created: int = 0
    foresights_created: int = 0


class PauseMemoryRequest(BaseModel):
    """Request to pause memory extraction for a file"""
    file_id: str


class PauseMemoryResponse(BaseModel):
    """Response from pause request"""
    success: bool
    file_id: str
    message: str


# Track files that should be paused (in-memory set for simplicity)
_paused_files: set[str] = set()


@router.post("/pause", response_model=PauseMemoryResponse)
async def pause_memory_for_file(request: PauseMemoryRequest) -> PauseMemoryResponse:
    """
    Pause memory extraction for a file.
    
    This marks the file as paused. If extraction is in progress, it will stop
    after the current chunk completes. The progress is saved and can be resumed
    with the extract endpoint.
    """
    from core.context import get_storage
    
    storage = get_storage()
    file_record = storage.get_file(request.file_id)
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_id}",
        )
    
    # Add to paused set (for in-progress extractions to check)
    _paused_files.add(request.file_id)
    
    # If currently extracting, mark as pending (paused)
    if file_record.memory_status == "extracting":
        file_record.memory_status = "pending"
        storage.upsert_file(file_record)
        return PauseMemoryResponse(
            success=True,
            file_id=request.file_id,
            message=f"Paused memory extraction for {file_record.name}. Progress saved at {file_record.memory_processed_chunks}/{file_record.memory_total_chunks} chunks.",
        )
    
    return PauseMemoryResponse(
        success=True,
        file_id=request.file_id,
        message=f"File {file_record.name} is not currently extracting (status: {file_record.memory_status})",
    )


@router.post("/memorize", response_model=MemorizeResult)
async def memorize(request: MemorizeRequest) -> MemorizeResult:
    """
    Process raw data and extract memories

    This endpoint processes conversation or other raw data to extract:
    - Episodic memories (summaries of events)
    - Event logs (atomic facts)
    - Foresights (prospective associations)
    - Profile updates
    """
    try:
        service = get_memory_service()
        return await service.memorize(request)
    except MemoryServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        ) from e


@router.post("/search", response_model=SearchMemoryResult)
async def search_memories(request: SearchMemoryRequest) -> SearchMemoryResult:
    """
    Search user memories

    Supports multiple retrieval methods:
    - keyword: BM25 keyword search
    - vector: Embedding-based semantic search
    - hybrid: Combined keyword + vector
    - rrf: Reciprocal Rank Fusion
    - agentic: LLM-guided multi-round retrieval
    """
    try:
        service = get_memory_service()
        return await service.search(request)
    except MemoryServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get("/{user_id}", response_model=UserMemorySummary)
async def get_user_memory_summary(user_id: str) -> UserMemorySummary:
    """
    Get summary of user's memories

    Returns profile information and memory counts.
    """
    try:
        service = get_memory_service()
        return await service.get_user_summary(user_id)
    except MemoryNotFound as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except MemoryServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get("/{user_id}/episodes", response_model=List[EpisodeRecord])
async def get_user_episodes(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> List[EpisodeRecord]:
    """
    Get user's episodic memories

    Episodic memories are narrative summaries of events and experiences.
    """
    try:
        service = get_memory_service()
        return await service.get_episodes(user_id, limit, offset)
    except MemoryServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get("/{user_id}/events", response_model=List[EventLogRecord])
async def get_user_event_logs(
    user_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> List[EventLogRecord]:
    """
    Get user's event logs (atomic facts)

    Event logs are fine-grained atomic facts extracted from episodic memories.
    """
    try:
        service = get_memory_service()
        return await service.get_event_logs(user_id, limit, offset)
    except MemoryServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get("/{user_id}/foresights", response_model=List[ForesightRecord])
async def get_user_foresights(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=200),
) -> List[ForesightRecord]:
    """
    Get user's foresights (prospective memories)

    Foresights are predictions and prospective associations extracted from memories.
    """
    try:
        service = get_memory_service()
        return await service.get_foresights(user_id, limit)
    except MemoryServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get("/{user_id}/memcells", response_model=List[MemCellRecord])
async def get_user_memcells(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> List[MemCellRecord]:
    """
    Get user's MemCells (source data for memory extraction)

    MemCells are the original data chunks that episodes and event logs are extracted from.
    """
    from core.context import get_storage
    from datetime import datetime

    storage = get_storage()
    records = storage.get_memcells(user_id, limit, offset)

    return [
        MemCellRecord(
            id=r.id,
            user_id=r.user_id,
            original_data=r.original_data,
            summary=r.summary,
            subject=r.subject,
            file_id=r.file_id,
            chunk_id=r.chunk_id,
            chunk_ordinal=r.chunk_ordinal,
            type=r.type,
            keywords=r.keywords,
            timestamp=datetime.fromisoformat(r.timestamp) if r.timestamp else datetime.now(),
            created_at=datetime.fromisoformat(r.created_at) if r.created_at else None,
            metadata=r.metadata,
        )
        for r in records
    ]


@router.get("/memcells/{memcell_id}", response_model=MemCellDetail)
async def get_memcell_detail(memcell_id: str) -> MemCellDetail:
    """
    Get a single MemCell with its linked episodes and event logs.
    """
    from core.context import get_storage
    from datetime import datetime

    storage = get_storage()
    record = storage.get_memcell(memcell_id)

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MemCell not found: {memcell_id}",
        )

    memcell = MemCellRecord(
        id=record.id,
        user_id=record.user_id,
        original_data=record.original_data,
        summary=record.summary,
        subject=record.subject,
        file_id=record.file_id,
        chunk_id=record.chunk_id,
        chunk_ordinal=record.chunk_ordinal,
        type=record.type,
        keywords=record.keywords,
        timestamp=datetime.fromisoformat(record.timestamp) if record.timestamp else datetime.now(),
        created_at=datetime.fromisoformat(record.created_at) if record.created_at else None,
        metadata=record.metadata,
    )

    # Get linked episodes
    episode_records = storage.get_episodes_by_memcell(memcell_id)
    episodes = [
        EpisodeRecord(
            id=ep.id,
            user_id=ep.user_id,
            summary=ep.summary,
            episode=ep.episode,
            subject=ep.subject,
            timestamp=datetime.fromisoformat(ep.timestamp) if ep.timestamp else datetime.now(),
            parent_memcell_id=ep.parent_memcell_id,
            metadata=ep.metadata,
        )
        for ep in episode_records
    ]

    # Get linked event logs (via episodes)
    event_logs = []
    for ep in episode_records:
        logs = storage.get_event_logs(record.user_id, limit=100, offset=0)
        for log in logs:
            if log.parent_episode_id == ep.id:
                event_logs.append(
                    EventLogRecord(
                        id=log.id,
                        user_id=log.user_id,
                        atomic_fact=log.atomic_fact,
                        timestamp=datetime.fromisoformat(log.timestamp) if log.timestamp else datetime.now(),
                        parent_episode_id=log.parent_episode_id,
                        metadata=log.metadata,
                    )
                )

    return MemCellDetail(
        memcell=memcell,
        episodes=episodes,
        event_logs=event_logs,
    )


@router.get("/memcells/by-file/{file_id}", response_model=List[MemCellRecord])
async def get_memcells_by_file(
    file_id: str,
    limit: int = Query(default=100, ge=1, le=500),
) -> List[MemCellRecord]:
    """
    Get MemCells for a specific file.
    """
    from core.context import get_storage
    from datetime import datetime

    storage = get_storage()
    records = storage.get_memcells_by_file(file_id, limit)

    return [
        MemCellRecord(
            id=r.id,
            user_id=r.user_id,
            original_data=r.original_data,
            summary=r.summary,
            subject=r.subject,
            file_id=r.file_id,
            chunk_id=r.chunk_id,
            chunk_ordinal=r.chunk_ordinal,
            type=r.type,
            keywords=r.keywords,
            timestamp=datetime.fromisoformat(r.timestamp) if r.timestamp else datetime.now(),
            created_at=datetime.fromisoformat(r.created_at) if r.created_at else None,
            metadata=r.metadata,
        )
        for r in records
    ]


@router.post("/extract", response_model=ExtractMemoryResponse)
async def extract_memory_for_file(request: ExtractMemoryRequest) -> ExtractMemoryResponse:
    """
    Manually trigger memory extraction for a specific file.
    
    This endpoint allows users to extract memories from a file on demand,
    regardless of the memory_extraction_stage setting.
    """
    import logging
    import datetime as dt
    import uuid
    
    from core.context import get_storage
    from services.memory.api_specs.memory_types import RawDataType, MemCell
    from services.memory.api_specs.memory_models import MemoryType
    from services.storage.memory import (
        MemCellRecord as StorageMemCellRecord,
        EpisodeRecord as StorageEpisodeRecord,
        EventLogRecord as StorageEventLogRecord,
        ForesightRecord as StorageForesightRecord,
    )
    import json
    
    logger = logging.getLogger(__name__)
    
    storage = get_storage()
    file_record = storage.get_file(request.file_id)
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_id}",
        )
    
    # Get file content from chunks
    all_chunks = storage.chunks_for_file(request.file_id)
    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No indexed content found for file: {file_record.name}. Please index the file first.",
        )
    
    # Prefer deep chunks (richer VLM content), fallback to fast chunks
    deep_chunks = [c for c in all_chunks if c.version == "deep"]
    fast_chunks = [c for c in all_chunks if c.version == "fast"]
    chunks = deep_chunks if deep_chunks else fast_chunks
    chunk_version = "deep" if deep_chunks else "fast"
    
    # Filter chunks with meaningful content (at least 100 chars)
    all_valid_chunks = [c for c in chunks if c.text and len(c.text) >= 100]
    
    if not all_valid_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No chunks with sufficient content for memory extraction (need at least 100 chars per chunk)",
        )
    
    # Custom chunk size: concat all text and re-chunk
    # Now supports breakpoint resume if using the same chunk_size
    use_custom_chunks = request.chunk_size is not None and request.chunk_size > 0
    custom_text_chunks: list[str] = []

    if use_custom_chunks:
        # Concatenate all chunk texts
        full_text = "\n\n".join(c.text for c in all_valid_chunks if c.text)

        # Split by custom chunk size with some overlap for context
        chunk_size = request.chunk_size
        overlap = min(200, chunk_size // 5)  # 20% overlap or 200 chars max

        pos = 0
        while pos < len(full_text):
            end = pos + chunk_size
            chunk_text = full_text[pos:end]
            if len(chunk_text) >= 100:  # Only include meaningful chunks
                custom_text_chunks.append(chunk_text)
            pos = end - overlap if end < len(full_text) else end

        total_valid_count = len(custom_text_chunks)

        # Check if we can resume from previous progress (same chunk_size)
        already_processed_count = 0
        if not request.force and file_record.memory_last_chunk_size == chunk_size:
            # Same chunk_size, can resume from where we left off
            already_processed_count = file_record.memory_processed_chunks or 0
            if already_processed_count > 0 and already_processed_count < total_valid_count:
                logger.info("🧠 [MEMORY] Resuming custom chunks: skipping %d already-processed, %d remaining",
                            already_processed_count, total_valid_count - already_processed_count)
                chunks_to_process_texts = custom_text_chunks[already_processed_count:]
            elif already_processed_count >= total_valid_count:
                # All chunks already processed
                file_record.memory_status = "extracted"
                file_record.memory_total_chunks = total_valid_count
                file_record.memory_processed_chunks = total_valid_count
                storage.upsert_file(file_record)
                return ExtractMemoryResponse(
                    success=True,
                    file_id=request.file_id,
                    message=f"All {already_processed_count} custom chunks already processed for {file_record.name}",
                    episodes_created=0,
                    event_logs_created=0,
                    foresights_created=0,
                )
            else:
                chunks_to_process_texts = custom_text_chunks
        elif not request.force and file_record.memory_last_chunk_size is not None and file_record.memory_last_chunk_size != chunk_size:
            # Different chunk_size, reset progress
            logger.info("🧠 [MEMORY] Chunk size changed (%d -> %d), restarting from beginning",
                        file_record.memory_last_chunk_size, chunk_size)
            already_processed_count = 0
            chunks_to_process_texts = custom_text_chunks
        else:
            chunks_to_process_texts = custom_text_chunks

        # Save the chunk_size used for this extraction
        file_record.memory_last_chunk_size = chunk_size

        logger.info("🧠 [MEMORY] Using custom chunk_size=%d, created %d chunks from %d original chunks (total %d chars)",
                    chunk_size, len(custom_text_chunks), len(all_valid_chunks), len(full_text))
    else:
        total_valid_count = len(all_valid_chunks)
        
        # Skip already-processed chunks unless force=True (断点续传)
        already_processed_count = 0
        if not request.force:
            already_processed_count = sum(1 for c in all_valid_chunks if c.memory_extracted_at is not None)
            pending_chunks = [c for c in all_valid_chunks if c.memory_extracted_at is None]
            if already_processed_count > 0:
                logger.info("🧠 [MEMORY] Resuming: skipping %d already-processed chunks, %d remaining", 
                            already_processed_count, len(pending_chunks))
            chunks_to_process = pending_chunks
            
            if not chunks_to_process:
                # All chunks already processed
                file_record.memory_status = "extracted"
                file_record.memory_total_chunks = total_valid_count
                file_record.memory_processed_chunks = total_valid_count
                storage.upsert_file(file_record)
                return ExtractMemoryResponse(
                    success=True,
                    file_id=request.file_id,
                    message=f"All {already_processed_count} chunks already processed for {file_record.name}",
                    episodes_created=0,
                    event_logs_created=0,
                    foresights_created=0,
                )
        else:
            chunks_to_process = all_valid_chunks
        
        chunks_to_process_texts = None  # Will use original chunks
    
    logger.info("🧠 [MEMORY] Manual extraction triggered for %s (%d/%d %s chunks%s%s)", 
                file_record.name, 
                len(custom_text_chunks) if use_custom_chunks else len(chunks_to_process), 
                total_valid_count, 
                "custom" if use_custom_chunks else chunk_version,
                ", force=True" if request.force else "",
                f", chunk_size={request.chunk_size}" if use_custom_chunks else "")
    
    # Set initial progress - mark as "extracting" with TOTAL chunks (not remaining)
    file_record.memory_status = "extracting"
    file_record.memory_total_chunks = total_valid_count
    file_record.memory_processed_chunks = already_processed_count  # Start from already processed count
    storage.upsert_file(file_record)
    
    try:
        service = get_memory_service()
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        
        memcells_created = 0
        episodes_created = 0
        event_logs_created = 0
        foresights_created = 0
        
        # Process each chunk as a separate MemCell
        # Determine iteration source based on mode
        if use_custom_chunks:
            iteration_items = list(enumerate(chunks_to_process_texts))
        else:
            iteration_items = list(enumerate(chunks_to_process))
        
        for chunk_idx, item in iteration_items:
            # Check if paused (user clicked pause button)
            if request.file_id in _paused_files:
                _paused_files.discard(request.file_id)  # Clear the pause flag
                file_record.memory_status = "pending"
                storage.upsert_file(file_record)
                logger.info("🧠 [MEMORY] Paused at chunk %d/%d", 
                           already_processed_count + chunk_idx, total_valid_count)
                return ExtractMemoryResponse(
                    success=True,
                    file_id=request.file_id,
                    message=f"Paused at {already_processed_count + chunk_idx}/{total_valid_count} chunks. Click 'Continue Memory' to resume.",
                    memcells_created=memcells_created,
                    episodes_created=episodes_created,
                    event_logs_created=event_logs_created,
                    foresights_created=foresights_created,
                )
            
            # Extract text and metadata based on mode
            if use_custom_chunks:
                chunk_text = item  # item is the text string
                chunk_id = f"custom_{chunk_idx}"
                chunk_ordinal = chunk_idx
                chunk_snippet = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
            else:
                chunk = item  # item is the chunk object
                chunk_text = chunk.text
                chunk_id = chunk.chunk_id
                chunk_ordinal = chunk.ordinal
                chunk_snippet = chunk.snippet
            
            current_progress = already_processed_count + chunk_idx + 1
            logger.info("🧠 [MEMORY] Processing chunk %d/%d (len=%d)", current_progress, total_valid_count, len(chunk_text))
            
            # Create MemCell for this chunk
            memcell_id = str(uuid.uuid4())
            original_data_dict = {
                "content": chunk_text,
                "chunk_id": chunk_id,
                "chunk_ordinal": chunk_ordinal,
                "file_name": file_record.name,
                "file_path": str(file_record.path) if file_record.path else "",
                "file_type": file_record.kind,
                "snippet": chunk_snippet,
            }

            memcell = MemCell(
                event_id=memcell_id,
                user_id_list=[request.user_id],
                original_data=[original_data_dict],
                timestamp=dt.datetime.now(dt.timezone.utc),
                summary=chunk_snippet or file_record.name,
                group_id=file_record.folder_id,
                participants=[],
                type=RawDataType.DOCUMENT,
            )

            # Persist MemCell to storage
            storage_memcell = StorageMemCellRecord(
                id=memcell_id,
                user_id=request.user_id,
                original_data=json.dumps([original_data_dict]),
                summary=chunk_snippet or file_record.name,
                subject=file_record.name,
                file_id=file_record.id,
                chunk_id=chunk_id,
                chunk_ordinal=chunk_ordinal,
                type="Document",
                keywords=None,
                timestamp=now,
                metadata={
                    "source": "manual_extract",
                    "file_path": str(file_record.path) if file_record.path else "",
                    "file_type": file_record.kind,
                }
            )
            storage.upsert_memcell(storage_memcell)
            memcells_created += 1
            logger.info("🧠 [MEMORY] Persisted MemCell %s for chunk %d", memcell_id, chunk_idx + 1)

            # Extract episodic memory for this chunk
            try:
                episode = await service.memory_manager.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.EPISODIC_MEMORY,
                    user_id=request.user_id,
                )
                
                if episode:
                    episode_id = str(uuid.uuid4())
                    storage_episode = StorageEpisodeRecord(
                        id=episode_id,
                        user_id=request.user_id,
                        summary=getattr(episode, "summary", chunk_snippet or ""),
                        episode=getattr(episode, "episode", ""),
                        subject=getattr(episode, "subject", file_record.name),
                        timestamp=now,
                        parent_memcell_id=memcell_id,
                        metadata={
                            "source": "manual_extract",
                            "file_id": file_record.id,
                            "file_name": file_record.name,
                            "chunk_id": chunk_id,
                            "chunk_ordinal": chunk_ordinal,
                            "custom_chunk_size": request.chunk_size if use_custom_chunks else None,
                        }
                    )
                    service.storage.upsert_episode(storage_episode)
                    episodes_created += 1
                    logger.info("🧠 [MEMORY] Created episode for chunk %d", chunk_idx + 1)
                    
                    # Extract event logs from this episode
                    try:
                        event_log = await service.memory_manager.extract_memory(
                            memcell=memcell,
                            memory_type=MemoryType.EVENT_LOG,
                            user_id=request.user_id,
                            episode_memory=episode,
                        )
                        if event_log:
                            facts = getattr(event_log, "atomic_fact", [])
                            if isinstance(facts, str):
                                facts = [facts]
                            for fact in facts:
                                if fact and fact.strip():
                                    log_id = str(uuid.uuid4())
                                    storage_log = StorageEventLogRecord(
                                        id=log_id,
                                        user_id=request.user_id,
                                        atomic_fact=fact.strip(),
                                        timestamp=now,
                                        parent_episode_id=episode_id,
                                        metadata={
                                            "source": "manual_extract",
                                            "file_id": file_record.id,
                                            "chunk_id": chunk_id,
                                        }
                                    )
                                    service.storage.upsert_event_log(storage_log)
                                    event_logs_created += 1
                    except Exception as e:
                        logger.warning("Event log extraction failed for chunk %d: %s", chunk_idx + 1, e)
                    
                    # Extract foresight from this episode
                    try:
                        foresight = await service.memory_manager.extract_memory(
                            memcell=memcell,
                            memory_type=MemoryType.FORESIGHT,
                            user_id=request.user_id,
                            episode_memory=episode,
                        )
                        if foresight:
                            foresight_text = getattr(foresight, "foresight", None)
                            if foresight_text and foresight_text.strip():
                                foresight_id = str(uuid.uuid4())
                                storage_foresight = StorageForesightRecord(
                                    id=foresight_id,
                                    user_id=request.user_id,
                                    foresight=foresight_text.strip(),
                                    evidence=getattr(foresight, "evidence", ""),
                                    start_time=getattr(foresight, "start_time", None),
                                    end_time=getattr(foresight, "end_time", None),
                                    timestamp=now,
                                    parent_episode_id=episode_id,
                                    metadata={
                                        "source": "manual_extract",
                                        "file_id": file_record.id,
                                        "chunk_id": chunk_id,
                                    }
                                )
                                service.storage.upsert_foresight(storage_foresight)
                                foresights_created += 1
                    except Exception as e:
                        logger.warning("Foresight extraction failed for chunk %d: %s", chunk_idx + 1, e)
            except Exception as e:
                logger.warning("Episode extraction failed for chunk %d: %s", chunk_idx + 1, e)
            
            # Mark original chunk as extracted (only for non-custom mode, for resume support)
            if not use_custom_chunks:
                storage.mark_chunk_memory_extracted(chunk.chunk_id)
            
            # Update progress after each chunk (whether successful or not)
            file_record.memory_processed_chunks = already_processed_count + chunk_idx + 1
            storage.upsert_file(file_record)
        
        processed_count = len(custom_text_chunks) if use_custom_chunks else len(chunks_to_process)
        logger.info("🧠 [MEMORY] Completed: %d memcells, %d episodes, %d event logs, %d foresights from %d chunks",
                    memcells_created, episodes_created, event_logs_created, foresights_created, processed_count)

        # Update file memory status
        file_record.memory_status = "extracted" if episodes_created > 0 else "skipped"
        file_record.memory_extracted_at = dt.datetime.now(dt.timezone.utc)
        file_record.memory_processed_chunks = total_valid_count  # Mark all as processed
        storage.upsert_file(file_record)

        return ExtractMemoryResponse(
            success=True,
            file_id=request.file_id,
            message=f"Memory extraction completed for {file_record.name}: processed {processed_count} {chunk_version} chunks (total: {total_valid_count})",
            memcells_created=memcells_created,
            episodes_created=episodes_created,
            event_logs_created=event_logs_created,
            foresights_created=foresights_created,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Memory extraction failed for %s: %s", file_record.name, e)
        import traceback
        traceback.print_exc()
        
        # Update file status to error
        file_record.memory_status = "error"
        storage.upsert_file(file_record)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory extraction failed: {str(e)}",
        ) from e


@router.get("/basic-profile/{user_id}/cached")
async def get_cached_basic_profile(user_id: str):
    """
    Get cached basic profile without regenerating.
    Returns 404 if no cached profile exists.
    """
    from services.memory.system_profile import load_basic_profile

    cached = load_basic_profile(user_id)
    if cached is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No cached profile found. Please analyze first.",
        )

    # Convert to response format
    topics = []
    for topic_data in cached.get("topics", []):
        subtopics = [
            ProfileSubtopic(
                name=s.get("name", ""),
                value=s.get("value"),
                confidence=s.get("confidence"),
                evidence=s.get("evidence"),
                description=s.get("description"),
            )
            for s in topic_data.get("subtopics", [])
        ]
        topics.append(ProfileTopic(
            topic_id=topic_data.get("topic_id", ""),
            topic_name=topic_data.get("topic_name", ""),
            icon=topic_data.get("icon"),
            subtopics=subtopics,
        ))

    return BasicProfileResponse(
        user_id=cached.get("user_id", user_id),
        user_name=cached.get("user_name"),
        topics=topics,
        raw_system_data=cached.get("raw_system_data"),
        scanned_at=cached.get("scanned_at", ""),
    )


@router.get("/basic-profile/{user_id}", response_model=BasicProfileResponse)
async def get_basic_profile(user_id: str) -> BasicProfileResponse:
    """
    Get LLM-inferred semantic profile from Mac system data.
    NOTE: This endpoint regenerates the profile. Use /cached for getting saved profile.

    Collects system information and uses LLM to infer a semantic user profile:
    - Collects: installed apps, dev tools, system preferences, user info
    - Infers: personality, interests, skills, work habits, goals, values

    The inference is done by LLM based on:
    - What applications reveal about profession, interests, lifestyle
    - Development tools indicate technical skills
    - System preferences suggest work habits and personal style
    - App combinations reveal potential roles
    """
    import logging
    from services.memory.system_profile import collect_and_infer_profile
    from core.context import get_llm_client

    logger = logging.getLogger(__name__)
    logger.info("Generating basic profile for user: %s", user_id)

    try:
        # Get LLM client
        llm_client = get_llm_client()

        # Collect system data and infer profile with LLM
        profile = await collect_and_infer_profile(user_id, llm_client)

        # Convert hierarchical topics
        topics = []
        for topic_data in profile.topics:
            subtopics = [
                ProfileSubtopic(
                    name=s.name,
                    value=s.value,
                    confidence=s.confidence,
                    evidence=s.evidence,
                    description=s.description,
                )
                for s in topic_data.subtopics
            ]
            topics.append(ProfileTopic(
                topic_id=topic_data.topic_id,
                topic_name=topic_data.topic_name,
                icon=topic_data.icon,
                subtopics=subtopics,
            ))

        # Convert to response model
        return BasicProfileResponse(
            user_id=profile.user_id,
            user_name=profile.user_name,
            topics=topics,
            # Legacy flat fields
            personality=profile.personality,
            interests=profile.interests,
            hard_skills=[
                SkillRecord(name=s.get("name", ""), level=s.get("level"))
                for s in profile.hard_skills
            ],
            soft_skills=[
                SkillRecord(name=s.get("name", ""), level=s.get("level"))
                for s in profile.soft_skills
            ],
            working_habit_preference=profile.working_habit_preference,
            user_goal=profile.user_goal,
            motivation_system=profile.motivation_system,
            value_system=profile.value_system,
            inferred_roles=profile.inferred_roles,
            raw_system_data=RawSystemDataRecord(
                username=profile.raw_system_data.get("username", ""),
                computer_name=profile.raw_system_data.get("computer_name", ""),
                shell=profile.raw_system_data.get("shell", ""),
                language=profile.raw_system_data.get("language", ""),
                region=profile.raw_system_data.get("region", ""),
                timezone=profile.raw_system_data.get("timezone", ""),
                appearance=profile.raw_system_data.get("appearance", ""),
                installed_apps=profile.raw_system_data.get("installed_apps", []),
                dev_tools=profile.raw_system_data.get("dev_tools", []),
            ) if profile.raw_system_data else None,
            scanned_at=profile.scanned_at,
        )
    except Exception as e:
        logger.error("Failed to generate basic profile: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate basic profile: {str(e)}",
        ) from e


@router.get("/basic-profile/{user_id}/stream")
async def stream_basic_profile(user_id: str):
    """
    Stream LLM-inferred semantic profile progressively using Server-Sent Events.

    This endpoint generates profile topics one by one, streaming each as it completes.
    The client receives events in this order:
    1. "init" - Initial data with raw system info
    2. "topic" - Each generated topic (basic_info, technical, work, etc.)
    3. "complete" - Generation finished
    4. "error" - If any topic fails (generation continues)

    Usage:
        const eventSource = new EventSource('/memory/basic-profile/user_id/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.type, data.data);
        };
    """
    import json
    import logging
    from fastapi.responses import StreamingResponse
    from services.memory.system_profile import generate_profile_progressive
    from core.context import get_llm_client

    logger = logging.getLogger(__name__)
    logger.info("Starting progressive profile stream for user: %s", user_id)

    async def event_generator():
        try:
            llm_client = get_llm_client()

            async for event in generate_profile_progressive(user_id, llm_client):
                # Format as SSE
                data = json.dumps(event, ensure_ascii=False)
                yield f"data: {data}\n\n"

        except Exception as e:
            logger.error("Stream error: %s", e)
            error_event = json.dumps({
                "type": "error",
                "data": {"error": str(e)}
            })
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
