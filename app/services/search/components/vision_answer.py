"""
Vision-based answering component.

Instead of using chunk text for answering, this component extracts page images
from the source documents and uses VLM (Vision Language Model) to answer questions
based on the visual content. This can be more accurate for documents with:
- Complex layouts (multi-column, tables, figures)
- Visual elements that are hard to extract as text
- Mathematical formulas or diagrams
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from services.llm.client import LlmClient
from core.config import settings

logger = logging.getLogger(__name__)


class VisionAnswerComponent:
    """
    Component that uses VLM to answer questions based on page images.

    Instead of processing chunk text, this retrieves the actual page images
    from source documents and sends them to the vision model for answering.
    """

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def _resolve_page_numbers(self, metadata: Optional[Dict[str, Any]]) -> List[int]:
        """
        Extract page numbers from chunk metadata.

        Chunks can have page info in various formats:
        - page_number: single page
        - page_numbers: list of pages
        - page_start/page_end: range of pages
        """
        if not metadata:
            return []

        pages: List[int] = []

        # Check for page_numbers list first
        page_numbers = metadata.get("page_numbers")
        if isinstance(page_numbers, list) and page_numbers:
            pages.extend([p for p in page_numbers if isinstance(p, int) and p > 0])
            return pages

        # Check for single page_number
        page_number = metadata.get("page_number") or metadata.get("page")
        if isinstance(page_number, int) and page_number > 0:
            return [page_number]

        # Check for page range
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        if isinstance(page_start, int) and isinstance(page_end, int):
            return list(range(page_start, page_end + 1))

        return []

    def _render_pdf_page(self, pdf_path: Path, page_num: int) -> Optional[bytes]:
        """
        Render a single PDF page to PNG bytes.

        Args:
            pdf_path: Path to the PDF file
            page_num: 1-indexed page number

        Returns:
            PNG image bytes, or None if rendering fails
        """
        if not fitz:
            logger.warning("PyMuPDF (fitz) not available, cannot render PDF pages")
            return None

        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return None

        try:
            doc = fitz.open(str(pdf_path))

            # Convert 1-indexed to 0-indexed
            page_index = page_num - 1
            if page_index < 0 or page_index >= len(doc):
                logger.warning(f"Page {page_num} out of range for {pdf_path} (has {len(doc)} pages)")
                doc.close()
                return None

            page = doc.load_page(page_index)

            # Render with zoom for better quality
            # For financial reports with tables, we need higher resolution
            # zoom=2.0 is 144 DPI, zoom=2.5 is 180 DPI
            target_zoom = 2.5
            min_zoom = 1.5  # Minimum ~108 DPI for readable tables
            max_pixels = settings.vision_max_pixels

            if max_pixels > 0:
                rect = page.rect
                if (rect.width * target_zoom) * (rect.height * target_zoom) > max_pixels:
                    zoom = max((max_pixels / (rect.width * rect.height)) ** 0.5, min_zoom)
                else:
                    zoom = target_zoom
            else:
                zoom = target_zoom

            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_bytes = pix.tobytes("png")

            doc.close()
            return img_bytes

        except Exception as e:
            logger.error(f"Failed to render PDF page {page_num} from {pdf_path}: {e}")
            return None

    def _calculate_max_image_pixels(self) -> int:
        """
        Calculate maximum image pixels based on context length budget.
        
        Formula: pixels ≈ tokens * 400 (Qwen VL uses ~1 token per 400 pixels)
        
        Context budget:
        - Total context: llm_context_tokens (e.g., 32768)
        - Reserved for prompt: ~500 tokens
        - Reserved for output: ~2000 tokens
        - Available for image: rest
        """
        context_tokens = settings.llm_context_tokens
        reserved_tokens = 2500  # prompt (~500) + output (~2000)
        available_image_tokens = context_tokens - reserved_tokens
        
        # Each token ≈ 400 pixels for Qwen VL
        pixels_from_context = available_image_tokens * 400
        
        # Also respect the explicit vision_max_pixels setting
        user_max_pixels = settings.vision_max_pixels
        
        # Use the smaller of the two limits
        if user_max_pixels > 0:
            max_pixels = min(pixels_from_context, user_max_pixels)
        else:
            max_pixels = pixels_from_context
        
        logger.debug(f"Image budget: context={context_tokens}, available_tokens={available_image_tokens}, max_pixels={max_pixels}")
        return max_pixels

    def _render_pdf_page_adaptive(self, pdf_path: Path, page_num: int) -> Optional[bytes]:
        """
        Render a PDF page with adaptive resolution based on:
        1. Page dimensions (wide pages get higher priority)
        2. Context length budget (don't exceed token limit)
        
        Args:
            pdf_path: Path to the PDF file
            page_num: 1-indexed page number
            
        Returns:
            PNG image bytes, or None if rendering fails
        """
        if not fitz:
            logger.warning("PyMuPDF (fitz) not available, cannot render PDF pages")
            return None

        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return None

        try:
            doc = fitz.open(str(pdf_path))

            page_index = page_num - 1
            if page_index < 0 or page_index >= len(doc):
                logger.warning(f"Page {page_num} out of range for {pdf_path}")
                doc.close()
                return None

            page = doc.load_page(page_index)
            rect = page.rect
            page_pixels = rect.width * rect.height
            
            # Calculate max pixels from context budget
            max_pixels = self._calculate_max_image_pixels()
            
            # Check if page is wide (landscape or two-column spread)
            is_wide = rect.width > rect.height * 1.2
            
            # For wide pages, we want higher resolution to read dense tables
            # But we must respect the context budget
            if is_wide:
                target_zoom = 3.0  # Higher zoom for wide pages (~216 DPI)
                min_zoom = 2.0    # Minimum ~144 DPI for readable tables
                logger.info(f"Page {page_num} is wide ({rect.width:.0f}x{rect.height:.0f})")
            else:
                target_zoom = 2.5  # Normal zoom for portrait pages (~180 DPI)
                min_zoom = 1.5    # Minimum ~108 DPI
            
            # Calculate actual zoom based on pixel budget
            target_pixels = page_pixels * (target_zoom ** 2)
            
            if target_pixels > max_pixels:
                # Scale down to fit budget, but respect minimum zoom
                budget_zoom = (max_pixels / page_pixels) ** 0.5
                zoom = max(budget_zoom, min_zoom)
                logger.info(f"Page {page_num}: target_zoom={target_zoom:.1f} exceeds budget, using zoom={zoom:.2f}")
            else:
                zoom = target_zoom

            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_bytes = pix.tobytes("png")
            
            actual_pixels = pix.width * pix.height
            estimated_tokens = actual_pixels // 400

            doc.close()
            logger.info(f"Rendered page {page_num}: zoom={zoom:.2f}, {pix.width}x{pix.height}={actual_pixels} pixels, ~{estimated_tokens} tokens")
            return img_bytes

        except Exception as e:
            logger.error(f"Failed to render PDF page {page_num} from {pdf_path}: {e}")
            return None

    async def process_chunk_with_vision(
        self,
        query: str,
        context_part: Dict[str, Any],
        file_path: Optional[Path] = None,
        file_name: Optional[str] = None,
        file_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single chunk using vision model on the source page.

        Instead of using chunk text, this renders the actual page image
        and sends it to VLM for answering.

        Args:
            query: The user's question
            context_part: Chunk data including metadata with page info
            file_path: Path to the source file (PDF)
            file_name: Name of the source file (for context)
            file_summary: Summary of the file (for context)

        Returns:
            Dict with answer result (same format as VerificationComponent)
        """
        metadata = context_part.get("metadata", {})
        source = context_part.get("source", "Unknown")
        index = context_part.get("index", 0)

        # Extract file context info
        if file_name is None:
            file_name = metadata.get("file_name") or metadata.get("name")
            if file_name is None and file_path:
                file_name = file_path.name
        if file_summary is None:
            file_summary = metadata.get("summary") or metadata.get("file_summary")

        logger.info(f"[VISION DEBUG] process_chunk_with_vision called, file_path={file_path}, source={source}, file_name={file_name}, metadata_keys={list(metadata.keys())}")

        # Resolve file path if not provided
        if file_path is None:
            path_str = metadata.get("path") or source
            if path_str:
                file_path = Path(path_str)

        # Check if this is a PDF
        if file_path is None or not str(file_path).lower().endswith(".pdf"):
            # Fall back to text-based processing for non-PDFs
            logger.info(f"[VISION DEBUG] Not a PDF or no file_path, falling back to text. file_path={file_path}")
            return await self._fallback_text_answer(query, context_part, file_name, file_summary)

        # Get page numbers from chunk metadata
        page_numbers = self._resolve_page_numbers(metadata)
        logger.info(f"[VISION DEBUG] Resolved page_numbers={page_numbers} from metadata={metadata}")

        if not page_numbers:
            # No page info available, fall back to text
            logger.info(f"[VISION DEBUG] No page info for chunk {index}, falling back to text")
            return await self._fallback_text_answer(query, context_part, file_name, file_summary)

        # Render the first page (most relevant)
        # Use adaptive rendering for better quality on wide/landscape pages
        page_num = page_numbers[0]
        logger.info(f"[VISION DEBUG] Rendering page {page_num} from {file_path}")
        page_image = self._render_pdf_page_adaptive(file_path, page_num)

        if page_image is None:
            logger.info(f"[VISION DEBUG] Failed to render page {page_num}, falling back to text")
            return await self._fallback_text_answer(query, context_part, file_name, file_summary)

        logger.info(f"[VISION DEBUG] Successfully rendered page {page_num}, image size={len(page_image)} bytes")

        # Use VLM to answer the question based on the page image
        try:
            # Build file context for the prompt
            file_context_parts = []
            if file_name:
                file_context_parts.append(f"Document: {file_name}")
            if file_summary:
                file_context_parts.append(f"Document Summary: {file_summary}")
            file_context_parts.append(f"Page: {page_num}")
            file_context = "\n".join(file_context_parts)

            system_prompt = (
                "You are a precise, fact-based assistant analyzing a document page.\n"
                "Answer the question based ONLY on what you can see in this page image.\n"
                "IMPORTANT: For tables, pay close attention to the units shown in headers or column titles "
                "(e.g., $'000, millions, %, etc.). Always include the correct unit in your answer.\n"
                "If the page does NOT contain information to answer the question, reply exactly: NO_ANSWER\n"
                "Do not make up or infer information that is not visible on the page.\n"
                "Keep your answer concise (1-2 sentences)."
            )

            user_prompt = f"{file_context}\n\nQuestion: {query}\n\nLook at the document page and answer the question."

            # Check if vision endpoint is configured
            if not settings.endpoints.vision:
                logger.warning("[VISION DEBUG] Vision endpoint not configured, falling back to text")
                return await self._fallback_text_answer(query, context_part, file_name, file_summary)

            logger.info(f"[VISION DEBUG] Calling VLM describe_frames with vision endpoint: {settings.endpoints.vision}")
            response = await self.llm_client.describe_frames(
                [page_image],
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=300,
            )

            response = response.strip()
            response_upper = response.upper()

            # Check for NO_ANSWER
            if "NO_ANSWER" in response_upper or "NO ANSWER" in response_upper:
                return {
                    "index": index,
                    "has_answer": False,
                    "content": "Vision model returned NO_ANSWER.",
                    "source": f"{source} (page {page_num})",
                    "confidence": 0.0,
                    "vision_processed": True,
                    "page_number": page_num,
                }

            # Check for other negative indicators
            if self._is_negative_response(response):
                return {
                    "index": index,
                    "has_answer": False,
                    "content": f"Negative response: {response[:100]}...",
                    "source": f"{source} (page {page_num})",
                    "confidence": 0.0,
                    "vision_processed": True,
                    "page_number": page_num,
                }

            # Valid answer found
            return {
                "index": index,
                "has_answer": True,
                "content": response,
                "source": f"{source} (page {page_num})",
                "confidence": 1.0,
                "vision_processed": True,
                "page_number": page_num,
            }

        except Exception as e:
            logger.error(f"Vision processing failed for chunk {index}: {e}")
            # Fall back to text-based processing
            return await self._fallback_text_answer(query, context_part, file_name, file_summary)

    def _is_negative_response(self, content: str) -> bool:
        """Check if a response indicates no answer found."""
        if not content:
            return True

        content_upper = content.upper().strip()
        negative_phrases = [
            "NO_ANSWER", "NO ANSWER",
            "I DON'T KNOW", "I DO NOT KNOW",
            "CANNOT FIND", "COULD NOT FIND",
            "NOT VISIBLE", "NOT SHOWN",
            "DOES NOT CONTAIN", "DOESN'T CONTAIN",
            "INFORMATION IS NOT", "INFORMATION NOT",
            "CANNOT ANSWER", "UNABLE TO ANSWER",
            "PAGE DOES NOT", "PAGE DOESN'T",
        ]

        for phrase in negative_phrases:
            if phrase in content_upper:
                return True

        return False

    async def _fallback_text_answer(
        self,
        query: str,
        context_part: Dict[str, Any],
        file_name: Optional[str] = None,
        file_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fall back to text-based answering when vision is not available.

        This uses the same logic as VerificationComponent.process_single_chunk.
        """
        system_prompt = (
            "You are a helpful assistant that extracts relevant information.\n"
            "Based on the context, answer the question in 1-2 sentences.\n"
            "If the context contains ANY related or partially relevant information, summarize what you found.\n"
            "Only reply NO_ANSWER if the context is completely unrelated to the question."
        )

        content = context_part.get("content", "")
        source = context_part.get("source", "Unknown")
        index = context_part.get("index", 0)
        metadata = context_part.get("metadata", {})

        # Extract file context info
        if file_name is None:
            file_name = metadata.get("file_name") or metadata.get("name")
        if file_summary is None:
            file_summary = metadata.get("summary") or metadata.get("file_summary")

        # Build file context for the prompt
        file_context_parts = []
        if file_name:
            file_context_parts.append(f"Document: {file_name}")
        if file_summary:
            file_context_parts.append(f"Document Summary: {file_summary}")
        file_context = "\n".join(file_context_parts)

        if file_context:
            user_prompt = f"{file_context}\n\nQuestion: {query}\n\nContext:\n{content}"
        else:
            user_prompt = f"Question: {query}\n\nContext:\n{content}"

        try:
            response = await self.llm_client.chat_complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
            )

            response = response.strip()
            response_upper = response.upper()

            if "NO_ANSWER" in response_upper or "NO ANSWER" in response_upper:
                return {
                    "index": index,
                    "has_answer": False,
                    "content": "Fallback text verification returned NO_ANSWER.",
                    "source": source,
                    "confidence": 0.0,
                    "vision_processed": False,
                }

            if not response or self._is_negative_response(response):
                return {
                    "index": index,
                    "has_answer": False,
                    "content": f"Negative response: {response[:50]}..." if response else "Empty response",
                    "source": source,
                    "confidence": 0.0,
                    "vision_processed": False,
                }

            return {
                "index": index,
                "has_answer": True,
                "content": response,
                "source": source,
                "confidence": 1.0,
                "vision_processed": False,
            }

        except Exception as e:
            logger.error(f"Fallback text processing failed for chunk {index}: {e}")
            return {
                "index": index,
                "has_answer": False,
                "content": f"Processing failed: {str(e)}",
                "source": source,
                "confidence": 0.0,
                "vision_processed": False,
            }

    async def process_chunks_with_vision(
        self,
        query: str,
        context_parts: List[Dict[str, Any]],
        storage: Any = None,  # IndexStorage for file path lookup
    ) -> List[Dict[str, Any]]:
        """
        Process multiple chunks using vision model.

        Deduplicates chunks from the same page to avoid redundant VLM calls.
        When multiple chunks come from the same (file_id, page_number), 
        only process the page once and share the result.

        Args:
            query: The user's question
            context_parts: List of chunk data
            storage: Optional storage for looking up file paths and file info

        Returns:
            List of answer results
        """
        # Step 1: Group chunks by (file_id, page_number) to deduplicate
        # Key: (file_id, page_num), Value: list of (original_index, part, file_info)
        page_groups: Dict[tuple, List[tuple]] = {}
        file_info_cache: Dict[str, tuple] = {}  # file_id -> (file_path, file_name, file_summary)

        for orig_idx, part in enumerate(context_parts):
            file_id = part.get("file_id")
            metadata = part.get("metadata", {})

            # Get page number from metadata
            page_numbers = self._resolve_page_numbers(metadata)
            page_num = page_numbers[0] if page_numbers else None

            # Get file info (cached)
            if file_id and file_id not in file_info_cache:
                file_path = None
                file_name = None
                file_summary = None
                if storage:
                    try:
                        file_record = storage.get_file(file_id)
                        if file_record:
                            if file_record.path:
                                file_path = Path(file_record.path)
                            file_name = file_record.name
                            file_summary = file_record.summary
                    except Exception as e:
                        logger.debug(f"Could not look up file {file_id}: {e}")

                # Fallback to source path
                if file_path is None:
                    source = part.get("source", "")
                    path_str = metadata.get("path") or source
                    if path_str and Path(path_str).exists():
                        file_path = Path(path_str)

                file_info_cache[file_id] = (file_path, file_name, file_summary)

            file_info = file_info_cache.get(file_id, (None, None, None))

            # Group by (file_id, page_num) - use None for non-PDF or unknown pages
            group_key = (file_id, page_num)
            if group_key not in page_groups:
                page_groups[group_key] = []
            page_groups[group_key].append((orig_idx, part, file_info))

        # Step 2: Process each unique (file_id, page) combination once
        # Cache results by group_key
        page_results_cache: Dict[tuple, Dict[str, Any]] = {}

        for group_key, group_items in page_groups.items():
            file_id, page_num = group_key

            # Use the first chunk as representative for this page
            orig_idx, representative_part, file_info = group_items[0]
            file_path, file_name, file_summary = file_info

            # Only process once per unique page
            if group_key not in page_results_cache:
                logger.info(f"[VISION DEDUP] Processing unique page: file_id={file_id}, page={page_num}, chunks_count={len(group_items)}")
                result = await self.process_chunk_with_vision(
                    query, representative_part, file_path, file_name, file_summary
                )
                page_results_cache[group_key] = result

        # Step 3: Build results list in original order, sharing results for same-page chunks
        results: List[Dict[str, Any]] = [None] * len(context_parts)  # type: ignore

        for group_key, group_items in page_groups.items():
            cached_result = page_results_cache[group_key]

            for orig_idx, part, _ in group_items:
                # Create a copy of the result with the correct index
                result_copy = cached_result.copy()
                result_copy["index"] = part.get("index", orig_idx)

                # Mark if this was a deduplicated result (not the first chunk for this page)
                if orig_idx != group_items[0][0]:
                    result_copy["deduplicated"] = True
                    result_copy["shared_with_index"] = group_items[0][1].get("index", group_items[0][0])

                results[orig_idx] = result_copy

        dedup_count = len(context_parts) - len(page_groups)
        if dedup_count > 0:
            logger.info(f"[VISION DEDUP] Deduplicated {dedup_count} chunks from {len(context_parts)} total (saved {dedup_count} VLM calls)")

        return results
