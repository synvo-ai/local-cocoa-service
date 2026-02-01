from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Callable, AsyncGenerator, Optional, TYPE_CHECKING
from core.config import settings
from services.search.types import EmbeddingUnavailableError
from services.search.progressive import extract_keywords_llm

if TYPE_CHECKING:
    from services.search.engine import SearchEngine
    from services.search.components.verification import VerificationComponent
    from services.search.components.vision_answer import VisionAnswerComponent

logger = logging.getLogger(__name__)

class StandardPipeline:
    def __init__(
        self, 
        engine: 'SearchEngine', 
        verification: 'VerificationComponent',
        vision_answer: Optional['VisionAnswerComponent'] = None,
    ):
        self.engine = engine
        self.verification = verification
        self.vision_answer = vision_answer

    def _dedupe_hits_by_page(self, hits: List[Any]) -> List[Any]:
        """
        Deduplicate hits by (file_id, page_number).
        For PDF files, multiple chunks from the same page are merged into one.
        Keeps the hit with the highest score for each unique page.
        Non-PDF files are not affected.
        """
        if not hits:
            return hits
        
        # Group hits by (file_id, page_num)
        # Key: (file_id, page_num or None), Value: best hit for that page
        page_best: Dict[tuple, Any] = {}
        non_pdf_hits: List[Any] = []
        
        for hit in hits:
            metadata = hit.metadata or {}
            file_id = hit.file_id
            
            # Get page number from metadata
            page_numbers = metadata.get("page_numbers", [])
            if not page_numbers:
                page_num = metadata.get("page_start") or metadata.get("page_number") or metadata.get("page")
                page_numbers = [page_num] if page_num else []
            
            # Check if this is a PDF (has page info)
            if page_numbers:
                page_num = page_numbers[0]
                key = (file_id, page_num)
                
                existing = page_best.get(key)
                if existing is None or (hit.score > existing.score):
                    page_best[key] = hit
            else:
                # Non-PDF or no page info - keep as is
                non_pdf_hits.append(hit)
        
        # Combine: deduplicated PDF hits + non-PDF hits
        deduped = list(page_best.values()) + non_pdf_hits
        
        # Sort by score descending
        deduped.sort(key=lambda h: h.score if hasattr(h, 'score') else 0, reverse=True)
        
        return deduped

    async def _process_hits_generator(
        self,
        query: str,
        hits: List[Any],
        step_generator: Callable[[], str],
        started_time: float,
        start_index: int = 1,
        use_vision_for_answer: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Helper generator to process a batch of hits.
        """
        def thinking_step(step_id, step_type, title, status="running", summary=None, details=None, hits=None, **kwargs):
            data = {
                "id": step_id,
                "type": step_type,
                "title": title,
                "status": status,
                "summary": summary,
                "details": details,
                "timestamp_ms": int((time.perf_counter() - started_time) * 1000), 
            }
            if hits:
                data["hits"] = hits
            data.update(kwargs)
            return {"type": "thinking_step", "data": data}

        if not hits:
            return

        # Deduplicate hits by (file_id, page_number) - keep only highest scoring hit per page
        # This reduces redundant processing when multiple chunks come from the same page
        deduped_hits = self._dedupe_hits_by_page(hits)
        logger.info(f"[PAGE DEDUP] Deduplicated {len(hits)} hits to {len(deduped_hits)} unique pages")

        # Inject indices
        for i, hit in enumerate(deduped_hits):
            if hit.metadata is None: hit.metadata = {}
            hit.metadata["index"] = start_index + i

        yield {"type": "hits", "data": [hit.model_dump(by_alias=True) for hit in deduped_hits]}
        yield {"type": "status", "data": "processing_chunks"}

        # Build context parts
        context_parts = []
        for i, hit in enumerate(deduped_hits):
             idx = start_index + i
             chunk_text = self.engine._chunk_text(hit)
             
             kind = hit.metadata.get("kind") if hit.metadata else None
             if kind == "image":
                parts = []
                if hit.summary: parts.append(f"[Image Description]: {hit.summary}")
                if chunk_text: parts.append(f"[Image Text]: {chunk_text}")
                snippet = "\n".join(parts)
                if not snippet: snippet = hit.snippet
             else:
                snippet = chunk_text or hit.snippet or hit.summary

             if snippet:
                source = hit.metadata.get("path") if hit.metadata else None
                label = source or hit.file_id
                
                # Enrich metadata with file_name and summary from file_record
                enriched_metadata = dict(hit.metadata) if hit.metadata else {}
                if hit.file_id and "file_name" not in enriched_metadata:
                    try:
                        file_record = self.engine.storage.get_file(hit.file_id)
                        if file_record:
                            enriched_metadata["file_name"] = file_record.name
                            if file_record.summary:
                                enriched_metadata["file_summary"] = file_record.summary
                    except Exception:
                        pass
                
                context_parts.append({
                    "index": idx,
                    "source": label,
                    "content": snippet[:settings.max_snippet_length],
                    "score": hit.score if hasattr(hit, 'score') else 0.0,
                    "file_id": hit.file_id,
                    "chunk_id": hit.chunk_id,
                    "metadata": enriched_metadata, 
                })

        # Relevance Filter
        filtered_parts = await self.verification.filter_relevant_chunks(query, context_parts)

        if len(filtered_parts) < len(context_parts):
            filtered_indices = {p.get("index") for p in filtered_parts}
            filtered_hits = [hit for i, hit in enumerate(deduped_hits) if (start_index + i) in filtered_indices]
            yield {"type": "hits", "data": [hit.model_dump(by_alias=True) for hit in filtered_hits]}
            
            rerank_step_id = step_generator()
            yield thinking_step(
                rerank_step_id, "analyze", "Relevance Filtering", "complete",
                f"Filtered to {len(filtered_parts)} relevant chunks",
                metadata={"resultsCount": len(filtered_parts), "relevantCount": len(filtered_parts)}
            )

        yield {"type": "status", "data": f"analyzing_{len(filtered_parts)}_chunks"}

        # Sequential Chunk Verification (with optional vision-based answering)
        sub_answers = []
        chunk_analysis = []
        high_quality_count = 0
        total_parts = len(filtered_parts)

        # For vision mode: build page deduplication cache to avoid processing same page multiple times
        vision_page_cache: Dict[tuple, Dict[str, Any]] = {}  # (file_id, page_num) -> result
        file_info_cache: Dict[str, tuple] = {}  # file_id -> (file_path, file_name, file_summary)

        for i, part in enumerate(filtered_parts):
            # Use vision-based answering if enabled and component is available
            logger.info(f"[VISION DEBUG] use_vision_for_answer={use_vision_for_answer}, vision_answer_component={self.vision_answer is not None}")
            if use_vision_for_answer and self.vision_answer is not None:
                file_id = part.get("file_id")
                metadata = part.get("metadata", {})
                
                # Get page number for deduplication
                page_numbers = metadata.get("page_numbers", [])
                if not page_numbers:
                    page_num = metadata.get("page_start") or metadata.get("page_number") or metadata.get("page")
                    page_numbers = [page_num] if page_num else []
                page_num = page_numbers[0] if page_numbers else None
                
                cache_key = (file_id, page_num)
                
                # Check if we already processed this page
                if cache_key in vision_page_cache:
                    # Reuse cached result
                    cached_result = vision_page_cache[cache_key]
                    result = cached_result.copy()
                    result["index"] = part.get("index", i)
                    result["deduplicated"] = True
                    logger.info(f"[VISION DEDUP] Reusing cached result for file_id={file_id}, page={page_num}")
                else:
                    # Get file info (cached)
                    if file_id and file_id not in file_info_cache:
                        file_path = None
                        file_name = None
                        file_summary = None
                        try:
                            file_record = self.engine.storage.get_file(file_id)
                            if file_record:
                                if file_record.path:
                                    file_path = Path(file_record.path)
                                file_name = file_record.name
                                file_summary = file_record.summary
                        except Exception as e:
                            logger.debug(f"Could not look up file {file_id}: {e}")
                        file_info_cache[file_id] = (file_path, file_name, file_summary)
                    
                    file_path, file_name, file_summary = file_info_cache.get(file_id, (None, None, None))
                    
                    result = await self.vision_answer.process_chunk_with_vision(
                        query, part, file_path, file_name, file_summary
                    )
                    
                    # Cache the result for this page
                    vision_page_cache[cache_key] = result
            else:
                result = await self.verification.process_single_chunk(query, part)
            
            sub_answers.append(result)
            if result.get("has_answer"):
                high_quality_count += 1
            
            anaysis_res = {
                "index": result.get("index"),
                "has_answer": result.get("has_answer", False),
                "comment": result.get("content", "") or None,
                "confidence": result.get("confidence", 0.0),
                "source": result.get("source", ""),
                "file_id": part.get("file_id", ""),
                "chunk_id": part.get("chunk_id", ""),
                "metadata": part.get("metadata"),
                "vision_processed": result.get("vision_processed", False),
                "deduplicated": result.get("deduplicated", False),
            }
            chunk_analysis.append(anaysis_res)

            yield {
                "type": "chunk_progress",
                "data": {
                    "processed_count": i + 1,
                    "total_count": total_parts,
                    "high_quality_count": high_quality_count,
                    "is_last": (i == total_parts - 1),
                    "current_file": part.get("source", "unknown"),
                    "chunk_result": anaysis_res,
                    "vision_mode": use_vision_for_answer,
                }
            }

        # Log deduplication stats
        if use_vision_for_answer and len(vision_page_cache) < total_parts:
            logger.info(f"[VISION DEDUP] Processed {len(vision_page_cache)} unique pages from {total_parts} chunks")

        yield {"type": "chunk_analysis", "data": chunk_analysis}
        yield {"type": "sub_answers", "data": sub_answers}


    async def execute(
        self,
        query: str,
        limit: int,
        step_generator: Callable[[], str],
        title_prefix: str = "",
        target_file_ids: List[str] | None = None,
        keywords: List[str] | None = None,
        rewritten_query: str | None = None,
        excluded_chunk_ids: set | None = None,
        global_start_index: int = 1,
        use_vision_for_answer: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the standard single-path search pipeline.
        Iterative approach: Keyword Search -> Verify -> (If needed) Semantic Search -> Verify.
        
        Args:
            rewritten_query: Pre-rewritten query optimized for embedding search (e.g., "NTU patents 2023")
        """
        started = time.perf_counter()
        
        def thinking_step(step_id, step_type, title, status="running", summary=None, details=None, hits=None, **kwargs):
            if title_prefix:
                title = f"{title_prefix}{title}"
            
            data = {
                "id": step_id,
                "type": step_type,
                "title": title,
                "status": status,
                "summary": summary,
                "details": details,
                "timestamp_ms": int((time.perf_counter() - started) * 1000),
            }
            if hits:
                data["hits"] = hits
            data.update(kwargs)
            return {"type": "thinking_step", "data": data}

        # Local set for chunks processed in THIS execution (to avoid re-processing semantic hits)
        local_processed_chunk_ids = set()
        # Include any globally excluded chunks from previous subqueries
        all_excluded = excluded_chunk_ids or set()
        total_good_answers = 0
        current_index = global_start_index  # Track the current global index
        logger.warning(f"[DEDUP DEBUG] StandardPipeline starting with global_start_index={global_start_index}, excluded_count={len(all_excluded)}")

        # Extract keywords using LLM if not provided or empty
        if not keywords:
            llm_client = getattr(self.engine, 'llm_client', None)
            keywords = await extract_keywords_llm(query, llm_client)
            logger.debug(f"LLM extracted keywords from query: {keywords}")
        else:
            logger.debug(f"Using pre-extracted keywords: {keywords}")

        # Step 1: Keyword Search (keywords already shown in Analyzing Query step)
        keyword_step_id = step_generator()
        keywords_display = ', '.join(keywords[:5]) if keywords else query
        yield thinking_step(
            keyword_step_id, "search", "Keyword Search", "running",
            f"Searching: {keywords_display}"
        )
        
        keyword_hits = []
        try:
            fts_query = " ".join(keywords) if keywords else query
            raw_keyword_hits = self.engine.storage.search_snippets_fts(fts_query, limit=limit * 2, file_ids=target_file_ids)
            # Filter out already-processed chunks from previous subqueries
            keyword_hits = [h for h in raw_keyword_hits if h.chunk_id not in all_excluded][:limit]
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")

        # Deduplicate keyword hits by page BEFORE emitting thinking_step
        # This ensures consistent index assignment between thinking_step and _process_hits_generator
        keyword_hits_deduped = self._dedupe_hits_by_page(keyword_hits) if keyword_hits else []
        logger.debug(f"Keyword hits: {len(keyword_hits)} -> {len(keyword_hits_deduped)} after page dedup")

        # Assign global indices to deduped hits BEFORE emitting thinking_step
        for i, hit in enumerate(keyword_hits_deduped):
            if hit.metadata is None:
                hit.metadata = {}
            hit.metadata["index"] = current_index + i

        hits_data = [h.model_dump(by_alias=True) for h in keyword_hits_deduped]
        yield thinking_step(
            keyword_step_id, "search", "Keyword Search", "complete",
            f"Found {len(keyword_hits_deduped)} matches via keyword search",
            hits=hits_data,
            metadata={"keywords": keywords}  # Include keywords in the step metadata
        )

        # Process Keyword Hits IMMEDIATELY (already deduped, so _process_hits_generator will just pass through)
        if keyword_hits_deduped:
            async for event in self._process_hits_generator(
                query, keyword_hits_deduped, step_generator, started, 
                start_index=current_index, use_vision_for_answer=use_vision_for_answer
            ):
                if event["type"] == "sub_answers":
                    # Count good answers to decide early stop
                    results = event["data"]
                    good_count = sum(1 for r in results if r.get("has_answer"))
                    total_good_answers += good_count
                    
                    # Track processed chunk IDs for semantic dedup within this execution
                    for h in keyword_hits_deduped:
                        local_processed_chunk_ids.add(h.chunk_id)
                
                yield event
            
            # Advance the global index counter by the ACTUAL deduped count
            current_index += len(keyword_hits_deduped)

        # Check Early Stop
        if total_good_answers >= 1:
             return

        # Not enough answers? Continue to Semantic Search.
        semantic_step_id = step_generator()
        
        # Use rewritten query for embedding if provided, otherwise fall back to original
        embedding_query = rewritten_query if rewritten_query else query
        yield thinking_step(
            semantic_step_id, "search", "Semantic Search", "running",
            f"Searching by meaning: {embedding_query[:50]}..." if len(embedding_query) > 50 else f"Searching by meaning: {embedding_query}"
        )
        
        semantic_hits = []
        try:
            # Use pre-rewritten query if available, otherwise do query expansion
            if rewritten_query:
                queries = [rewritten_query, query]  # Use both rewritten and original
            else:
                rewrite = await self.engine._maybe_rewrite_query(query)
                queries = rewrite.variants(include_original=True, limit=4)
            embeddings = await self.engine.embedding_client.encode(queries)
            
            if embeddings:
                semantic_hits, _ = self.engine._collect_multi_vector_hits(
                    queries[:len(embeddings)], embeddings, limit, file_ids=target_file_ids
                )
        except EmbeddingUnavailableError:
             yield thinking_step(semantic_step_id, "search", "Semantic Search", "error", "Embedding service unavailable")
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        # Filter Semantic Hits (exclude already processed in this execution AND globally excluded)
        combined_exclusions = local_processed_chunk_ids | all_excluded
        unique_semantic_hits = [h for h in semantic_hits if h.chunk_id not in combined_exclusions]
        
        # Deduplicate semantic hits by page BEFORE emitting thinking_step
        semantic_hits_deduped = self._dedupe_hits_by_page(unique_semantic_hits) if unique_semantic_hits else []
        logger.debug(f"Semantic hits: {len(unique_semantic_hits)} -> {len(semantic_hits_deduped)} after page dedup")

        # Assign global indices to deduped semantic hits BEFORE emitting thinking_step
        for i, hit in enumerate(semantic_hits_deduped):
            if hit.metadata is None:
                hit.metadata = {}
            hit.metadata["index"] = current_index + i

        hits_data = [h.model_dump(by_alias=True) for h in semantic_hits_deduped]
        yield thinking_step(
            semantic_step_id, "search", "Semantic Search", "complete",
            f"Found {len(semantic_hits_deduped)} new matches via Vector Search",
            hits=hits_data
        )

        if not semantic_hits_deduped:
             if total_good_answers == 0:
                 yield {"type": "status", "data": "no_results"}
                 yield {"type": "done_internal", "data": "No matching files found."}
             return

        # Process Semantic Hits (already deduped)
        # Continue index from global counter
        async for event in self._process_hits_generator(
            query, semantic_hits_deduped, step_generator, started, 
            start_index=current_index, use_vision_for_answer=use_vision_for_answer
        ):
             yield event
        
        # Note: current_index is not returned, but caller tracks via len(hits) returned
