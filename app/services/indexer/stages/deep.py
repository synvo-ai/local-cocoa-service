"""Deep Processor - Round 2.

Uses VLM (Vision Language Model) for deeper understanding of visual content.
Produces additional chunks (v2) that complement fast text extraction.
VLM text extraction + embedding happen together since VLM is already slow.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from typing import Optional

from services.chunker import ChunkingPipeline, chunking_pipeline
from core.config import settings
from core.models import ChunkSnapshot, FileRecord, VectorDocument
from services.llm.client import EmbeddingClient, LlmClient
from services.storage import IndexStorage
from services.vlm import VisionProcessor
from core.vector_store import VectorStore, get_vector_store
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


class DeepProcessor:
    """Round 2: Deep vision-based processing.

    Responsibilities:
    - Use VLM to extract richer descriptions from images/PDFs
    - Generate chunks_v2 (deep version)
    - Generate embeddings for deep chunks
    - Store in both SQLite and Qdrant
    - Update deep_stage from 0 -> 2 (text+embed together)

    Only processes files that would benefit from VLM:
    - Images
    - PDFs with visual content
    - Presentations with images

    Skips:
    - Plain text files
    - Audio/video (handled separately)
    - Files already processed by deep
    """

    def __init__(
        self,
        storage: IndexStorage,
        state_manager: StateManager,
        *,
        embedding_client: EmbeddingClient,
        llm_client: LlmClient,
        chunker: ChunkingPipeline = chunking_pipeline,
        vectors: Optional[VectorStore] = None,
        vision_processor: Optional[VisionProcessor] = None,
        enable_memory_extraction: bool = True,
        memory_user_id: str = "default_user",
    ) -> None:
        self.storage = storage
        self.state_manager = state_manager
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.chunker = chunker
        self.vector_store = vectors or get_vector_store()
        self.vision_processor = vision_processor or VisionProcessor(llm_client)
        self.enable_memory_extraction = enable_memory_extraction
        self.memory_user_id = memory_user_id

    async def process(self, file_id: str, file_record: Optional[FileRecord] = None) -> bool:
        """Process a single file with VLM for deep understanding.

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
            logger.debug("File not found (may have been pruned): %s", file_id)
            return False

        # Must have completed fast round first
        if file_record.fast_stage < 2:
            logger.warning("File %s hasn't completed fast round yet", file_id)
            return False

        if file_record.deep_stage >= 2:
            logger.debug("File %s already has deep_stage >= 2, skipping", file_id)
            return True

        if file_record.deep_stage == -2:
            logger.debug("File %s was marked as skipped for deep", file_id)
            return True

        # Check if this file type benefits from VLM
        if not self._should_process_deep(file_record):
            logger.info("Skipping deep processing for %s (not suitable)", file_record.name)
            self.storage.update_file_stage(file_id, deep_stage=-2)  # -2 = skipped
            return True

        # Set active file for progress display
        self.state_manager.set_active_file(
            folder_id=file_record.folder_id,
            file_path=file_record.path,
            file_name=file_record.name,
            kind=file_record.kind,
        )

        path = file_record.path
        if not path.exists():
            reason = f"File no longer exists: {path}"
            logger.warning(reason)
            self.storage.update_file_stage(file_id, deep_stage=-1)
            self.storage.mark_file_error(file_id, reason)
            return False

        try:
            self.state_manager.set_active_stage(
                stage="deep_vision",
                detail=f"Deep processing {path.name}",
                progress=0.0,
                event=f"VLM analyzing {path.name}"
            )

            # Get deep text based on file type
            deep_text = None
            deep_chunks: list[ChunkSnapshot] = []

            if file_record.kind == "image":
                deep_text = await self._process_image(file_record)
            elif file_record.kind == "document" and file_record.extension == "pdf":
                deep_text, deep_chunks = await self._process_pdf(file_record)
            elif file_record.kind == "presentation":
                deep_text = await self._process_presentation(file_record)

            if not deep_text and not deep_chunks:
                logger.info("No deep content extracted for %s", file_record.name)
                now = dt.datetime.now(dt.timezone.utc)
                self.storage.update_file_stage(
                    file_id,
                    deep_stage=2,
                    deep_text_at=now,
                    deep_embed_at=now
                )
                return True

            # If we got text but no chunks, build chunks
            if deep_text and not deep_chunks:
                deep_chunks = self._build_deep_chunks(file_record, deep_text)

            # Set version to "deep" for all chunks
            for chunk in deep_chunks:
                chunk.version = "deep"

            # Store chunks in SQLite
            self.storage.replace_chunks(file_id, deep_chunks, version="deep")

            # Generate embeddings for deep chunks
            now = dt.datetime.now(dt.timezone.utc)
            if deep_chunks:
                self.state_manager.set_active_stage(
                    stage="deep_embed",
                    detail=f"Embedding {len(deep_chunks)} deep chunks",
                    progress=50.0,
                    event=f"Embedding {len(deep_chunks)} deep chunks",
                )

                vectors = await self._embed_chunks(deep_chunks)

                self.state_manager.set_active_stage(
                    stage="deep_embed",
                    detail=f"Storing {len(deep_chunks)} vectors",
                    progress=80.0,
                    event=f"Embedding complete, storing vectors",
                )

                # Store in vector database
                documents: list[VectorDocument] = []
                for chunk, vector in zip(deep_chunks, vectors):
                    doc_metadata = {
                        "chunk_id": chunk.chunk_id,
                        "file_id": file_record.id,
                        "file_name": file_record.name,
                        "path": str(file_record.path),
                        "folder_id": file_record.folder_id,
                        "extension": file_record.extension,
                        "kind": file_record.kind,
                        "snippet": chunk.snippet,
                        "version": "deep",
                        # Privacy level for filtering - external requests cannot see private files
                        "privacy_level": file_record.privacy_level,
                    }
                    if chunk.metadata:
                        for key in ["page_number", "page_numbers"]:
                            if key in chunk.metadata:
                                doc_metadata[key] = chunk.metadata[key]

                    documents.append(VectorDocument(
                        doc_id=chunk.chunk_id,
                        vector=vector,
                        metadata=doc_metadata,
                    ))

                if documents:
                    try:
                        self.vector_store.upsert(documents)
                        self.vector_store.flush()
                    except Exception as exc:
                        logger.warning("Vector store upsert failed for deep chunks: %s", exc)

                # Update file metadata
                file_record.metadata = file_record.metadata or {}
                file_record.metadata["vector_chunks_deep"] = [d.doc_id for d in documents]
                file_record.metadata["chunk_count_deep"] = len(deep_chunks)
                file_record.metadata["deep_processed"] = True

            # Update stage
            file_record.deep_stage = 2
            file_record.deep_text_at = now
            file_record.deep_embed_at = now
            self.storage.upsert_file(file_record)

            logger.info(
                "Deep processing completed for %s: %d chunks",
                file_record.name, len(deep_chunks)
            )

            self.state_manager.set_active_stage(
                stage="deep_complete",
                detail=f"Deep processing complete: {len(deep_chunks)} chunks",
                progress=100.0,
                event=f"✓ {file_record.name} done ({len(deep_chunks)} chunks)",
            )

            # Memory extraction during deep stage (if configured)
            # Combine fast text + deep text for richer memory extraction
            combined_text = ""
            fast_chunks = [
                c for c in self.storage.chunks_for_file(file_id)
                if getattr(c, "version", "fast") == "fast"
            ]
            if fast_chunks:
                combined_text = "\n\n".join(c.text for c in fast_chunks if c.text)
            if deep_text:
                if combined_text:
                    combined_text += "\n\n--- Deep Analysis ---\n\n" + deep_text
                else:
                    combined_text = deep_text

            should_extract_memory = (
                self.enable_memory_extraction
                and MEMORY_AVAILABLE
                and combined_text
                and settings.memory_extraction_stage == "deep"
            )
            logger.info(
                "🧠 Memory check (deep): enable=%s, available=%s, text_len=%d, stage=%s, will_extract=%s",
                self.enable_memory_extraction, MEMORY_AVAILABLE,
                len(combined_text) if combined_text else 0,
                settings.memory_extraction_stage, should_extract_memory
            )
            if should_extract_memory:
                logger.info("🧠 Triggering memory extraction for %s (deep stage)", file_record.name)
                asyncio.create_task(
                    self._extract_memory_safe(file_record, combined_text),
                    name=f"memory_extract_deep_{file_id}"
                )

            return True

        except Exception as exc:
            reason = f"Deep processing failed: {type(exc).__name__}: {exc}"
            logger.error(
                "Deep processing failed for %s (%s): %s",
                file_id, file_record.name if file_record else file_id, exc,
                exc_info=True,
            )
            self.storage.update_file_stage(file_id, deep_stage=-1)
            self.storage.mark_file_error(file_id, reason)
            return False

        finally:
            self.state_manager.reset_active_state()

    def _should_process_deep(self, file_record: FileRecord) -> bool:
        """Determine if a file would benefit from VLM processing."""
        # Images always benefit from VLM
        if file_record.kind == "image":
            return True

        # PDFs with pages benefit from VLM
        if file_record.kind == "document" and file_record.extension == "pdf":
            # Check if we have page images or preview
            if file_record.preview_image:
                return True
            page_count = file_record.page_count or 0
            if page_count > 0:
                return True

        # Presentations with images
        if file_record.kind == "presentation":
            return True

        # Skip text-only files
        if file_record.kind in ("document",) and file_record.extension in ("txt", "md", "csv"):
            return False

        # Skip audio/video (handled separately)
        if file_record.kind in ("audio", "video"):
            return False

        return False

    async def _process_image(self, record: FileRecord) -> Optional[str]:
        """Process image using VLM."""
        if not record.preview_image:
            # Try to read the image file
            try:
                with open(record.path, "rb") as f:
                    image_bytes = f.read()
            except Exception:
                return None
        else:
            image_bytes = record.preview_image

        try:
            text = await self.vision_processor.process_image(
                image_bytes,
                mode="deep",
                prompt=prompts.IMAGE_PROMPT
            )
            return text
        except Exception as e:
            logger.warning("VLM processing failed for image %s: %s", record.path, e)
            return None

    @staticmethod
    def _render_pdf_pages(path: Path, zoom: float = 2.0) -> dict[str, bytes]:
        """Render PDF pages to PNG bytes using PyMuPDF — fast, no OCR/VLM."""
        import fitz  # PyMuPDF

        results: dict[str, bytes] = {}
        doc = fitz.open(str(path))
        try:
            matrix = fitz.Matrix(zoom, zoom)
            for page_index in range(len(doc)):
                page = doc[page_index]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                results[f"page_{page_index + 1}"] = pix.tobytes("png")
        finally:
            doc.close()
        return results

    async def _process_pdf(self, record: FileRecord) -> tuple[Optional[str], list[ChunkSnapshot]]:
        """Process PDF pages using VLM."""
        # Show parsing phase in UI
        self.state_manager.set_active_stage(
            stage="deep_vision",
            detail=f"Rendering PDF pages for {record.name}",
            progress=1.0,
            event=f"Rendering PDF pages: {record.name}",
        )

        try:
            # Lightweight page render — only converts PDF pages to PNG bytes.
            # Previous code used content_router.parse(indexing_mode="deep") which
            # ran OCR + VLM on every page (taking minutes), then _process_pdf
            # discarded that text and ran VLM again on the same images.
            page_images = await asyncio.to_thread(
                self._render_pdf_pages, record.path
            )
        except Exception as e:
            logger.warning("Failed to render PDF pages for deep processing: %s", e)
            return None, []

        if not page_images:
            return None, []

        sorted_pages = sorted(page_images.items(), key=lambda x: int(x[0].split("_")[1]))
        total_pages = len(sorted_pages)
        page_results: list[str] = []
        chunks: list[ChunkSnapshot] = []
        now = dt.datetime.now(dt.timezone.utc)

        logger.info("VLM processing %d pages for %s", total_pages, record.name)
        self.state_manager.set_active_stage(
            stage="deep_vision",
            detail=f"VLM processing page 1/{total_pages}",
            step_current=1,
            step_total=total_pages,
            progress=2.0,
            event=f"Starting VLM for {total_pages} pages",
        )

        for i, (page_key, image_bytes) in enumerate(sorted_pages):
            # Update detail/step BEFORE VLM call so UI shows which page is active
            self.state_manager.set_active_stage(
                stage="deep_vision",
                detail=f"VLM processing page {i + 1}/{total_pages}",
                step_current=i + 1,
                step_total=total_pages,
            )

            page_num = int(page_key.split("_")[1])

            if settings.vision_batch_delay_ms > 0 and i > 0:
                await asyncio.sleep(settings.vision_batch_delay_ms / 1000)

            try:
                result = await self.vision_processor.process_image(
                    image_bytes,
                    mode="deep",
                    prompt=prompts.PDF_PAGE_PROMPT
                )

                cleaned = (result or "").strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```\w*\s+|\s+```$", "", cleaned, flags=re.MULTILINE).strip()

                if cleaned:
                    page_results.append(cleaned)

                    # Create a chunk for this page
                    chunk_id = f"{record.id}::deep::page_{page_num}"
                    chunks.append(ChunkSnapshot(
                        chunk_id=chunk_id,
                        file_id=record.id,
                        ordinal=page_num - 1,
                        text=cleaned,
                        snippet=cleaned[:400],
                        token_count=max(len(cleaned) // 4, 1),
                        char_count=len(cleaned),
                        section_path=f"page_{page_num}",
                        metadata={
                            "page_number": page_num,
                            "page_numbers": [page_num],
                            "source": "vlm",
                        },
                        created_at=now,
                        version="deep",
                    ))

            except Exception as e:
                logger.warning("VLM failed for page %d of %s: %s", page_num, record.path, e)

            # Update progress AFTER each page completes (2%-50% range for VLM)
            page_progress = 2.0 + ((i + 1) / max(total_pages, 1)) * 48.0
            self.state_manager.set_active_stage(
                stage="deep_vision",
                detail=f"Completed page {i + 1}/{total_pages}",
                step_current=i + 1,
                step_total=total_pages,
                progress=page_progress,
                event=f"Page {i + 1}/{total_pages} done",
            )

        combined_text = "\n\n".join(page_results) if page_results else None
        return combined_text, chunks

    async def _process_presentation(self, record: FileRecord) -> Optional[str]:
        """Process presentation slides using VLM."""
        # Similar to image processing
        if record.preview_image:
            try:
                text = await self.vision_processor.process_image(
                    record.preview_image,
                    mode="deep",
                    prompt="Describe this presentation slide in detail."
                )
                return text
            except Exception as e:
                logger.warning("VLM processing failed for presentation %s: %s", record.path, e)
        return None

    def _build_deep_chunks(self, record: FileRecord, text: str) -> list[ChunkSnapshot]:
        """Build chunks from deep text extraction."""
        if not text or not text.strip():
            return []

        now = dt.datetime.now(dt.timezone.utc)
        chunk_id = f"{record.id}::deep::full"

        return [ChunkSnapshot(
            chunk_id=chunk_id,
            file_id=record.id,
            ordinal=0,
            text=text,
            snippet=text[:400],
            token_count=max(len(text) // 4, 1),
            char_count=len(text),
            section_path=None,
            metadata={"source": "vlm"},
            created_at=now,
            version="deep",
        )]

    async def _embed_chunks(self, chunks: list[ChunkSnapshot]) -> list[list[float]]:
        """Generate embeddings for chunks."""
        texts = [c.text.strip()[:settings.embed_max_chars] for c in chunks if c.text.strip()]
        if not texts:
            return []

        batch_size = max(settings.embed_batch_size, 1)
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response_vectors = await self.embedding_client.encode(batch)
            vectors.extend(response_vectors)

        return vectors

    # --- Memory Extraction (Deep Stage) ---

    async def _extract_memory_safe(self, record: FileRecord, text: str) -> None:
        """Safe wrapper for memory extraction - catches all exceptions."""
        print(f"🧠 [MEMORY-DEEP] _extract_memory_safe called for {record.name}")
        logger.info("🧠 _extract_memory_safe (deep) called for %s", record.name)
        try:
            await self._extract_memory(record, text)
            print(f"🧠 [MEMORY-DEEP] _extract_memory completed for {record.name}")
        except Exception as exc:
            print(f"🧠 [MEMORY-DEEP] Memory extraction failed for {record.path}: {exc}")
            logger.warning("Memory extraction (deep) failed for %s: %s", record.path, exc)
            import traceback
            traceback.print_exc()
            # Don't re-raise - this runs in background

    async def _extract_memory(self, record: FileRecord, text: str) -> None:
        """Extract memory from file content during deep stage.

        This extracts:
        - Episode memory (what happened)
        - Event logs (actionable events)  
        - Foresights (future actions/predictions)
        """
        if not MEMORY_AVAILABLE:
            return

        print(f"🧠 [MEMORY-DEEP] _extract_memory started for {record.name}")
        memory_service = get_memory_service(self.storage)

        # Create RawData from file content
        raw_data = RawData(
            raw_data_id=str(uuid.uuid4()),
            raw_data_type=RawDataType.DOCUMENT,
            author_id=self.memory_user_id,
            content=text[:50000],  # Limit content size
            content_description=f"Deep indexed file: {record.name}",
            created_at=record.indexed_at.isoformat() if record.indexed_at else dt.datetime.now(dt.timezone.utc).isoformat(),
        )

        try:
            # Extract memcell (boundary detection)
            memcell, status = await memory_service.memory_manager.extract_memcell(
                history_raw_data_list=[],
                new_raw_data_list=[raw_data],
                raw_data_type=RawDataType.DOCUMENT,
                group_id=record.folder_id,
                group_name=str(record.path.parent.name) if record.path else None,
                user_id_list=[self.memory_user_id],
            )

            if not memcell:
                logger.info("No memcell extracted for %s", record.name)
                return

            # Extract episode memory
            print(f"🧠 [MEMORY-DEEP] Calling extract_memory for EPISODIC_MEMORY...")
            episode = await memory_service.memory_manager.extract_memory(
                memcell=memcell,
                memory_type=MemoryType.EPISODIC_MEMORY,
                user_id=self.memory_user_id,
                group_id=record.folder_id,
            )

            if episode:
                # Save episode to storage
                episode_record = StorageEpisodeRecord(
                    id=str(uuid.uuid4()),
                    file_id=record.id,
                    user_id=self.memory_user_id,
                    description=episode.description if hasattr(episode, 'description') else str(episode),
                    happened_at=episode.happened_at if hasattr(episode, 'happened_at') else dt.datetime.now(dt.timezone.utc),
                    created_at=dt.datetime.now(dt.timezone.utc),
                    metadata={"source": "deep_index", "file_name": record.name},
                )
                self.storage.upsert_episode(episode_record)

                # Extract event logs from episode
                try:
                    event_log = await memory_service.memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.EVENT_LOG,
                        user_id=self.memory_user_id,
                        episode_memory=episode,
                    )
                    if event_log:
                        event_record = StorageEventLogRecord(
                            id=str(uuid.uuid4()),
                            file_id=record.id,
                            user_id=self.memory_user_id,
                            event_type=event_log.event_type if hasattr(event_log, 'event_type') else "document_event",
                            description=event_log.description if hasattr(event_log, 'description') else str(event_log),
                            trigger_time=event_log.trigger_time if hasattr(event_log, 'trigger_time') else None,
                            created_at=dt.datetime.now(dt.timezone.utc),
                            metadata={"source": "deep_index"},
                        )
                        self.storage.upsert_event_log(event_record)
                except Exception as e:
                    logger.debug("Event log extraction skipped: %s", e)

                # Extract foresights from episode
                try:
                    foresight_list = await memory_service.memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.FORESIGHT,
                        user_id=self.memory_user_id,
                        episode_memory=episode,
                    )
                    if foresight_list:
                        for foresight in (foresight_list if isinstance(foresight_list, list) else [foresight_list]):
                            foresight_record = StorageForesightRecord(
                                id=str(uuid.uuid4()),
                                file_id=record.id,
                                user_id=self.memory_user_id,
                                action=foresight.action if hasattr(foresight, 'action') else str(foresight),
                                predicted_time=foresight.predicted_time if hasattr(foresight, 'predicted_time') else None,
                                created_at=dt.datetime.now(dt.timezone.utc),
                                metadata={"source": "deep_index"},
                            )
                            self.storage.upsert_foresight(foresight_record)
                except Exception as e:
                    logger.debug("Foresight extraction skipped: %s", e)

            logger.info(
                "Memory extraction (deep) completed for %s",
                record.name
            )

            # Update file metadata
            record.metadata = record.metadata or {}
            record.metadata["memory_extracted"] = True
            record.metadata["memory_extraction_stage"] = "deep"
            record.memory_status = "extracted"
            record.memory_extracted_at = dt.datetime.now(dt.timezone.utc)
            self.storage.upsert_file(record)
        except Exception as exc:
            logger.warning("Memory extraction (deep) failed for %s: %s", record.path, exc)
            raise
