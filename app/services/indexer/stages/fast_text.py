"""Fast Text Processor - Round 1 Stage 1.

Extracts text from files using fast methods (PyMuPDF, OCR).
Produces text chunks that enable keyword search immediately.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from services.chunker import ChunkingPipeline, chunking_pipeline
from services.parser.base import TextBlockInfo
from core.config import settings
from core.content import ContentRouter, content_router
from core.models import (
    ChunkSnapshot,
    FileRecord,
    FolderRecord,
    IngestArtifact,
    infer_kind,
)
from services.llm.client import LlmClient
from services.storage import IndexStorage
from ..scanner import fingerprint  # Note: checksum removed for performance
from ..state import StateManager
from .. import prompts

logger = logging.getLogger(__name__)

# Memory integration (optional)
try:
    from services.memory.service import get_memory_service
    from services.memory.api_specs.dtos.memory_command import RawData
    from services.memory.api_specs.memory_types import RawDataType, MemoryType
    from services.storage.memory import (
        EpisodeRecord as StorageEpisodeRecord,
        EventLogRecord as StorageEventLogRecord,
        ForesightRecord as StorageForesightRecord,
    )
    import uuid
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


class FastTextProcessor:
    """Round 1 Stage 1: Fast text extraction.

    Responsibilities:
    - Extract text from files using fast methods (PyMuPDF OCR, text extraction)
    - Generate text chunks (v1)
    - Store chunks in SQLite (FTS5 for keyword search)
    - Update fast_stage from 0 -> 1

    Does NOT:
    - Generate embeddings (that's FastEmbedProcessor)
    - Use VLM vision models (that's DeepProcessor)
    """

    def __init__(
        self,
        storage: IndexStorage,
        state_manager: StateManager,
        *,
        content: ContentRouter = content_router,
        chunker: ChunkingPipeline = chunking_pipeline,
        llm_client: Optional[LlmClient] = None,
        enable_memory_extraction: bool = True,
        memory_user_id: str = "default_user",
    ) -> None:
        self.storage = storage
        self.state_manager = state_manager
        self.content_router = content
        self.chunker = chunker
        self.llm_client = llm_client or LlmClient()
        self.enable_memory_extraction = enable_memory_extraction
        self.memory_user_id = memory_user_id

    async def process(self, file_id: str, file_record: Optional[FileRecord] = None) -> bool:
        """Process a single file: extract text and create chunks.

        Args:
            file_id: The file ID to process
            file_record: Optional pre-fetched file record to avoid redundant DB lookup

        Returns:
            True if successful, False otherwise
        """
        if file_record is None:
            file_record = self.storage.get_file(file_id)
        if not file_record:
            # File may have been deleted between listing and processing (race condition)
            # This is expected behavior during concurrent indexing/pruning
            logger.debug("File not found (may have been pruned): %s", file_id)
            return False

        if file_record.fast_stage >= 1:
            logger.debug("File %s already has fast_stage >= 1, skipping", file_id)
            return True

        path = file_record.path
        if not path.exists():
            reason = f"File no longer exists: {path}"
            logger.warning(reason)
            self.storage.update_file_stage(file_id, fast_stage=-1)
            self.storage.mark_file_error(file_id, reason)
            return False

        try:
            # Update state for UI
            self.state_manager.set_active_stage(
                stage="fast_text",
                detail=f"Extracting text from {path.name}",
                progress=0.0,
                event=f"Processing {path.name}"
            )

            # Parse file content using fast mode
            parsed = await asyncio.to_thread(
                self.content_router.parse, path, indexing_mode="fast"
            )

            # Build metadata
            stat = path.stat()
            extension = path.suffix.lower().lstrip(".")
            mime_type, _ = mimetypes.guess_type(str(path))

            # OPTIMIZATION: Skip expensive SHA256 checksum during fast indexing.
            # Use a lightweight fingerprint (size + mtime) for change detection.
            # Full checksum can be computed lazily during deep indexing if needed.
            # This saves ~30-50% of processing time for large files.
            lightweight_checksum = f"size:{stat.st_size}:mtime:{int(stat.st_mtime)}"

            # Update file record with parsed content
            file_record.mime_type = mime_type
            file_record.checksum_sha256 = lightweight_checksum  # Lightweight checksum for fast mode
            file_record.duration_seconds = parsed.duration_seconds
            file_record.page_count = parsed.page_count
            file_record.preview_image = parsed.preview_image

            # Merge metadata
            file_record.metadata = file_record.metadata or {}
            file_record.metadata.update(parsed.metadata)
            file_record.metadata["file_name"] = file_record.name
            file_record.metadata["name"] = file_record.name
            file_record.metadata["path"] = str(file_record.path)
            file_record.metadata["full_path"] = str(file_record.path)
            file_record.metadata["folder_id"] = file_record.folder_id
            file_record.metadata["extension"] = file_record.extension
            file_record.metadata["size"] = file_record.size
            file_record.metadata["kind"] = file_record.kind

            # Create artifact for chunking
            artifact = IngestArtifact(
                record=file_record,
                text=parsed.text,
                chunks=[],
                page_mapping=parsed.page_mapping
            )

            # Build chunks (pass text_blocks for spatial metadata)
            chunks = self._build_chunks(file_record, artifact, text_blocks=parsed.text_blocks)

            # Set chunk version to "fast"
            for chunk in chunks:
                chunk.version = "fast"

            # Store chunks in SQLite (FTS5 will auto-sync)
            self.storage.replace_chunks(file_id, chunks, version="fast")

            # Update file record
            file_record.metadata["vector_chunks_fast"] = [c.chunk_id for c in chunks]
            file_record.metadata["chunk_count_fast"] = len(chunks)
            file_record.metadata["chunk_strategy"] = "fast_text_v1"

            # Note: Summary generation is deferred to batch processing
            # after all files complete fast_text stage.
            # Store the text length for later summary generation
            file_record.metadata["text_length"] = len(parsed.text) if parsed.text else 0

            # Update stage
            now = dt.datetime.now(dt.timezone.utc)
            file_record.fast_stage = 1
            file_record.fast_text_at = now
            file_record.index_status = "pending"  # Still pending until embed done

            self.storage.upsert_file(file_record)

            logger.info(
                "Fast text completed for %s: %d chunks created",
                path.name, len(chunks)
            )

            # Trigger memory extraction in background (non-blocking)
            # Only extract during fast stage if memory_extraction_stage == "fast"
            should_extract_memory = (
                self.enable_memory_extraction
                and MEMORY_AVAILABLE
                and artifact.text
                and settings.memory_extraction_stage == "fast"
            )
            logger.info(
                "🧠 Memory check: enable=%s, available=%s, text_len=%d, stage=%s, will_extract=%s",
                self.enable_memory_extraction, MEMORY_AVAILABLE,
                len(artifact.text) if artifact.text else 0,
                settings.memory_extraction_stage, should_extract_memory
            )
            if should_extract_memory:
                logger.info("🧠 Triggering memory extraction for %s (fast stage)", path.name)
                asyncio.create_task(
                    self._extract_memory_safe(file_record, artifact.text),
                    name=f"memory_extract_{file_id}"
                )

            return True

        except Exception as exc:
            reason = f"Fast text failed: {type(exc).__name__}: {exc}"
            logger.error(
                "Fast text failed for %s (%s): %s",
                file_id, file_record.name if file_record else file_id, exc,
                exc_info=True,
            )
            self.storage.update_file_stage(file_id, fast_stage=-1)
            self.storage.mark_file_error(file_id, reason)
            return False

        finally:
            self.state_manager.reset_active_state()

    def _calculate_max_input_chars(self) -> int:
        """Dynamically calculate max input chars based on LLM context length.

        Reserves tokens for:
        - System prompt (~100 tokens)
        - File metadata (~50 tokens)
        - Output tokens (summary_max_tokens)
        - Safety margin (10%)

        Returns:
            Maximum number of characters for document content
        """
        context_tokens = getattr(settings, "llm_context_tokens", 32768)
        chars_per_token = getattr(settings, "llm_chars_per_token", 8)
        summary_max_tokens = getattr(settings, "summary_max_tokens", 256)

        # Reserve tokens for system prompt, metadata, and output
        reserved_tokens = 100 + 50 + summary_max_tokens
        # Apply 10% safety margin
        available_tokens = int((context_tokens - reserved_tokens) * 0.9)

        max_chars = available_tokens * chars_per_token

        # Also respect the explicit setting if it's lower
        explicit_limit = getattr(settings, "summary_input_max_chars", 100000)

        return min(max_chars, explicit_limit)

    async def _generate_llm_summary(
        self,
        record: FileRecord,
        text: str,
    ) -> str:
        """Generate a summary using LLM instead of simple text truncation.

        Args:
            record: The file record with metadata
            text: The full text content of the file

        Returns:
            A concise summary of the file content
        """
        if not text or not text.strip():
            return ""

        # Prepare metadata context
        metadata_block = self._format_file_metadata_for_llm(record)

        # Dynamically calculate max input chars based on context length
        max_input_chars = self._calculate_max_input_chars()
        truncated_content = text.strip()[:max_input_chars]
        if len(text) > max_input_chars:
            truncated_content += "\n...[content truncated]..."

        prompt = f"{metadata_block}\n\nContent:\n{truncated_content}"

        try:
            messages = [
                {"role": "system", "content": prompts.DEFAULT_SUMMARY_PROMPT},
                {"role": "user", "content": prompt},
            ]
            summary = await self.llm_client.chat_complete(
                messages,
                max_tokens=max(int(getattr(settings, "summary_max_tokens", 256)), 32),
                temperature=0.2,
            )
            cleaned = (summary or "").strip()
            if cleaned:
                logger.info("LLM summary generated for %s: %d chars", record.name, len(cleaned))
                return cleaned
        except Exception as exc:
            logger.warning("LLM summarization failed for %s: %s, falling back to truncation", record.path, exc)

        # Fallback: use first 500 chars if LLM fails
        return text.strip()[:500]

    def _format_file_metadata_for_llm(self, record: FileRecord) -> str:
        """Format file metadata as context for LLM summarization."""
        modified = record.modified_at.isoformat() if record.modified_at else "unknown"
        created = record.created_at.isoformat() if record.created_at else "unknown"
        size = str(record.size) if record.size is not None else "unknown"
        return "\n".join([
            f"File name: {record.name}",
            f"Path: {record.path}",
            f"Kind: {record.kind}",
            f"Extension: {record.extension}",
            f"Size bytes: {size}",
            f"Modified at: {modified}",
            f"Created at: {created}",
        ])

    async def generate_summaries_batch(
        self,
        folder_id: Optional[str] = None,
        batch_size: int = 50,
    ) -> int:
        """Generate LLM summaries for all files that have completed fast_text but lack summaries.

        This is called after all files complete fast_text stage to batch process summaries.

        Args:
            folder_id: Optional folder to limit processing to
            batch_size: Number of files to process in each batch

        Returns:
            Number of files successfully summarized
        """
        success_count = 0

        # Get files that need summary (fast_stage >= 1 and no summary)
        files_needing_summary = self.storage.list_files_needing_summary(
            folder_id=folder_id,
            limit=batch_size,
        )

        if not files_needing_summary:
            logger.info("No files need summary generation")
            return 0

        logger.info("Generating summaries for %d files", len(files_needing_summary))

        for file_record in files_needing_summary:
            try:
                # Update state for UI
                self.state_manager.set_active_stage(
                    stage="fast_summary",
                    detail=f"Generating summary for {file_record.name}",
                    progress=success_count / len(files_needing_summary),
                    event=f"Summarizing {file_record.name}"
                )

                # Get the file's text content from chunks
                chunks = self.storage.chunks_for_file(file_record.id)
                if not chunks:
                    # No chunks found - this file needs to be re-indexed
                    # Reset fast_stage to 0 so it will be re-processed
                    logger.warning(
                        "No chunks found for %s (id=%s), resetting fast_stage to 0 for re-indexing",
                        file_record.name, file_record.id
                    )
                    self.storage.update_file_stage(file_record.id, fast_stage=0)
                    continue

                # Concatenate chunk texts to reconstruct file content
                text = "\n".join(c.text for c in chunks if c.text)

                if not text.strip():
                    logger.warning("No text content in chunks for %s, skipping summary", file_record.name)
                    continue

                # Generate summary
                summary = await self._generate_llm_summary(file_record, text)

                if summary:
                    file_record.summary = summary
                    self.storage.upsert_file(file_record)
                    success_count += 1
                    logger.info("Summary generated for %s", file_record.name)

            except Exception as exc:
                logger.warning("Failed to generate summary for %s: %s", file_record.path, exc)
                continue

        self.state_manager.reset_active_state()
        logger.info("Batch summary generation complete: %d/%d files", success_count, len(files_needing_summary))

        return success_count

    def _build_chunks(
        self,
        record: FileRecord,
        artifact: IngestArtifact,
        text_blocks=None,
    ) -> list[ChunkSnapshot]:
        """Build text chunks from the artifact.

        Args:
            record: The file record
            artifact: The ingest artifact containing text and metadata
            text_blocks: Optional list of TextBlockInfo with spatial data for chunk area visualization
        """
        text = artifact.text
        if not text or not text.strip():
            return []

        now = dt.datetime.now(dt.timezone.utc)
        chunk_tokens = settings.rag_chunk_size

        # For PDF: chunk each page separately to ensure page numbers are correct
        # This prevents content from different pages being mixed in one chunk
        if record.kind == "document" and record.extension == "pdf":
            page_texts = record.metadata.get("page_texts") or []
            if page_texts and isinstance(page_texts, list):
                return self._build_pdf_chunks_by_page(
                    record, page_texts, chunk_tokens, now, text_blocks=text_blocks
                )

        # For non-PDF files: use standard chunking
        overlap_tokens = settings.rag_chunk_overlap

        payloads = self.chunker.build(
            record.id,
            text,
            page_mapping=artifact.page_mapping,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            text_blocks=text_blocks,
        )

        from core.models import SourceRegion

        snapshots = []
        for payload in payloads:
            # Convert source_regions from chunker types to model types
            source_regions = None
            if payload.source_regions:
                source_regions = [
                    SourceRegion(
                        page_num=r.page_num,
                        bbox=list(r.bbox),
                        confidence=r.confidence,
                    )
                    for r in payload.source_regions
                ]

            snapshots.append(ChunkSnapshot(
                chunk_id=payload.chunk_id,
                file_id=record.id,
                ordinal=payload.ordinal,
                text=payload.text,
                snippet=payload.snippet,
                token_count=payload.token_count,
                char_count=payload.char_count,
                section_path=payload.section_path,
                metadata=payload.metadata,
                created_at=now,
                version="fast",
                page_num=payload.page_num,
                bbox=list(payload.bbox) if payload.bbox else None,
                source_regions=source_regions,
            ))

        if not snapshots and text.strip():
            # Fallback: single chunk for entire text
            fallback_id = f"{record.id}::fast::full"
            snapshots.append(ChunkSnapshot(
                chunk_id=fallback_id,
                file_id=record.id,
                ordinal=0,
                text=text,
                snippet=text[:400],
                token_count=max(len(text) // 4, 1),
                char_count=len(text),
                section_path=None,
                metadata={},
                created_at=now,
                version="fast",
            ))

        return snapshots

    def _build_pdf_chunks_by_page(
        self,
        record: FileRecord,
        page_texts: list[str],
        chunk_tokens: int,
        now: dt.datetime,
        text_blocks=None,
    ) -> list[ChunkSnapshot]:
        """
        Build chunks for PDF by processing each page separately.
        This ensures page numbers are always correct - no cross-page content mixing.

        Args:
            record: The file record
            page_texts: List of text content for each page
            chunk_tokens: Target chunk size in tokens
            now: Current timestamp
            text_blocks: Optional list of TextBlockInfo with spatial data
        """
        from core.models import SourceRegion

        snapshots: list[ChunkSnapshot] = []
        ordinal = 0

        # Group text_blocks by page for efficient lookup
        blocks_by_page = {}
        if text_blocks:
            for block in text_blocks:
                page = block.page_num
                if page not in blocks_by_page:
                    blocks_by_page[page] = []
                blocks_by_page[page].append(block)

        for page_num, page_text in enumerate(page_texts, start=1):
            page_text = page_text or ""
            if not page_text.strip():
                continue

            # Build page-level page_mapping (single page)
            page_mapping = [(0, len(page_text), page_num)]

            # Get text_blocks for this page, adjusted to page-relative offsets.
            # Text blocks have absolute char offsets (from full document), but the
            # chunker receives only page_text (0-based), so we must rebase offsets.
            page_blocks = blocks_by_page.get(page_num)
            if page_blocks:
                page_base = page_blocks[0].char_start  # first block's absolute offset
                page_blocks = [
                    TextBlockInfo(
                        text=b.text,
                        page_num=b.page_num,
                        bbox=b.bbox,
                        char_start=b.char_start - page_base,
                        char_end=b.char_end - page_base,
                        confidence=b.confidence,
                        block_type=b.block_type,
                    )
                    for b in page_blocks
                ]

            # Chunk this page with no overlap
            payloads = self.chunker.build(
                record.id,
                page_text,
                page_mapping=page_mapping,
                chunk_tokens=chunk_tokens,
                overlap_tokens=0,  # No overlap for PDF to keep page boundaries clean
                text_blocks=page_blocks,
            )

            for payload in payloads:
                # Override chunk_id to include page number for uniqueness
                chunk_id = f"{record.id}::fast::p{page_num}::{ordinal}"

                # Ensure page metadata is correct
                metadata = payload.metadata.copy()
                metadata["page_numbers"] = [page_num]
                metadata["page_start"] = page_num
                metadata["page_end"] = page_num

                # Convert source_regions from chunker types to model types
                source_regions = None
                if payload.source_regions:
                    source_regions = [
                        SourceRegion(
                            page_num=r.page_num,
                            bbox=list(r.bbox),
                            confidence=r.confidence,
                        )
                        for r in payload.source_regions
                    ]

                snapshots.append(ChunkSnapshot(
                    chunk_id=chunk_id,
                    file_id=record.id,
                    ordinal=ordinal,
                    text=payload.text,
                    snippet=payload.snippet,
                    token_count=payload.token_count,
                    char_count=payload.char_count,
                    section_path=payload.section_path,
                    metadata=metadata,
                    created_at=now,
                    version="fast",
                    page_num=payload.page_num or page_num,
                    bbox=list(payload.bbox) if payload.bbox else None,
                    source_regions=source_regions,
                ))
                ordinal += 1

        return snapshots

    # --- Memory Extraction ---

    async def _extract_memory_safe(self, record: FileRecord, text: str) -> None:
        """Safe wrapper for memory extraction - catches all exceptions."""
        print(f"🧠 [MEMORY] _extract_memory_safe called for {record.name}")
        logger.info("🧠 _extract_memory_safe called for %s", record.name)
        try:
            await self._extract_memory(record, text)
            print(f"🧠 [MEMORY] _extract_memory completed for {record.name}")
        except Exception as exc:
            print(f"🧠 [MEMORY] Memory extraction failed for {record.path}: {exc}")
            logger.warning("Memory extraction failed for %s: %s", record.path, exc)
            import traceback
            traceback.print_exc()
            # Don't re-raise - this runs in background

    async def _extract_memory(self, record: FileRecord, text: str) -> None:
        """
        Extract memories from file content and save to database.
        Runs in background - does not block file indexing.
        """
        print(f"🧠 [MEMORY] _extract_memory started for {record.name}")

        if not MEMORY_AVAILABLE:
            print("🧠 [MEMORY] MEMORY_AVAILABLE is False, skipping")
            return

        if not text or len(text.strip()) < 100:
            print(f"🧠 [MEMORY] Text too short ({len(text.strip()) if text else 0} chars), skipping")
            return

        print(f"🧠 [MEMORY] Text length: {len(text)} chars")

        # Check if LLM is configured and available (use same endpoint as app)
        llm_url = settings.endpoints.llm_url
        print(f"🧠 [MEMORY] Checking LLM at {llm_url}")
        try:
            import aiohttp
            # Try health endpoint, fallback to just checking if server responds
            health_url = f"{llm_url.rstrip('/')}/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    print(f"🧠 [MEMORY] LLM health check: {resp.status}")
        except Exception as e:
            print(f"🧠 [MEMORY] LLM not reachable: {e}, skipping")
            return

        try:
            memory_service = get_memory_service()
            now = dt.datetime.now(dt.timezone.utc).isoformat()

            # For documents, we bypass the conversation boundary detection and directly
            # create a MemCell. The ConvMemCellExtractor is designed for conversations
            # that accumulate messages over time - when history_raw_data_list is empty,
            # it returns should_end=False (waiting for more messages), which means
            # no MemCell is created for single documents.
            #
            # Instead, we directly create a MemCell for the document content.
            from services.memory.api_specs.memory_types import MemCell
            import uuid

            # Create MemCell directly for document
            # For documents, original_data contains the text content and file metadata
            # This gives the LLM more context about the document
            memcell = MemCell(
                event_id=str(uuid.uuid4()),
                user_id_list=[self.memory_user_id],
                original_data=[{
                    "content": text[:50000],  # The indexed text content
                    "file_name": record.name,
                    "file_path": str(record.path),
                    "file_type": record.kind,
                    "file_extension": record.extension,
                    "file_size_bytes": record.size,
                    "page_count": record.page_count,
                    "modified_at": record.modified_at.isoformat() if record.modified_at else None,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                    "summary": record.summary,
                }],
                timestamp=dt.datetime.now(dt.timezone.utc),
                summary=record.summary or record.name,
                group_id=None,
                participants=[],
                type=RawDataType.DOCUMENT,
            )

            print(f"🧠 [MEMORY] Created MemCell: event_id={memcell.event_id}")

            episodes_created = 0
            event_logs_created = 0
            foresights_created = 0

            # Extract episodic memory
            print(f"🧠 [MEMORY] Calling extract_memory for EPISODIC_MEMORY...")
            episode = await memory_service.memory_manager.extract_memory(
                memcell=memcell,
                memory_type=MemoryType.EPISODIC_MEMORY,
                user_id=self.memory_user_id,
            )
            print(f"🧠 [MEMORY] Episode result: {episode is not None}")

            episode_id = None
            if episode:
                episode_id = str(uuid.uuid4())
                storage_episode = StorageEpisodeRecord(
                    id=episode_id,
                    user_id=self.memory_user_id,
                    summary=getattr(episode, "summary", record.summary or ""),
                    episode=getattr(episode, "episode", ""),
                    subject=getattr(episode, "subject", record.name),
                    timestamp=now,
                    metadata={
                        "source": "file_indexer",
                        "file_id": record.id,
                        "file_name": record.name,
                        "file_path": str(record.path),
                    }
                )
                memory_service.storage.upsert_episode(storage_episode)
                episodes_created = 1
                logger.info("Created episode %s for file %s", episode_id, record.name)

                # Extract event logs
                try:
                    event_log = await memory_service.memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.EVENT_LOG,
                        user_id=self.memory_user_id,
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
                                    user_id=self.memory_user_id,
                                    atomic_fact=fact.strip(),
                                    timestamp=now,
                                    parent_episode_id=episode_id,
                                )
                                memory_service.storage.upsert_event_log(storage_log)
                                event_logs_created += 1
                except Exception as e:
                    logger.warning("Event log extraction failed for %s: %s", record.path, e)

                # Extract foresights
                try:
                    foresight_list = await memory_service.memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.FORESIGHT,
                        user_id=self.memory_user_id,
                        episode_memory=episode,
                    )
                    if foresight_list:
                        items = foresight_list if isinstance(foresight_list, list) else [foresight_list]
                        for foresight in items:
                            content = getattr(foresight, "foresight", "") or getattr(foresight, "content", "")
                            if content and content.strip():
                                fs_id = str(uuid.uuid4())
                                storage_foresight = StorageForesightRecord(
                                    id=fs_id,
                                    user_id=self.memory_user_id,
                                    content=content.strip(),
                                    evidence=getattr(foresight, "evidence", None),
                                    parent_episode_id=episode_id,
                                )
                                memory_service.storage.upsert_foresight(storage_foresight)
                                foresights_created += 1
                except Exception as e:
                    logger.warning("Foresight extraction failed for %s: %s", record.path, e)

            logger.info(
                "Memory extraction completed for %s: episodes=%d, events=%d, foresights=%d",
                record.name, episodes_created, event_logs_created, foresights_created
            )

            # Update file metadata and memory status
            record.metadata = record.metadata or {}
            record.metadata["memory_extracted"] = True
            record.metadata["memory_episode_id"] = episode_id
            record.metadata["memory_events_count"] = event_logs_created
            record.metadata["memory_foresights_count"] = foresights_created
            record.memory_status = "extracted"
            record.memory_extracted_at = dt.datetime.now(dt.timezone.utc)
            self.storage.upsert_file(record)

        except Exception as exc:
            logger.warning("Memory extraction failed for %s: %s", record.path, exc)
            import traceback
            traceback.print_exc()
            # Update memory status to error
            try:
                record.memory_status = "error"
                self.storage.upsert_file(record)
            except Exception:
                pass  # Ignore storage errors during error handling
