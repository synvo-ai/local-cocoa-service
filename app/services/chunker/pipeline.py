from __future__ import annotations

import datetime as dt
import os
from typing import List, Optional, Tuple, TYPE_CHECKING

import xxhash  # type: ignore[import]

try:
    import tiktoken  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]

from .types import ChunkPayload, SourceRegion
from .semantic import SemanticSplitter, LIST_PATTERN

if TYPE_CHECKING:
    from services.parser.base import TextBlockInfo


class ChunkingPipeline:
    """
    Produces hierarchical chunks with token metrics for downstream retrieval.
    
    This implementation uses semantic-aware chunking that:
    1. Prioritizes paragraph boundaries
    2. Respects sentence boundaries when splitting is needed
    3. Preserves lists, code blocks, and tables as complete units
    4. Merges small segments and splits large ones intelligently
    """

    def __init__(
        self,
        *,
        embedding_model: str = "text-embedding-3-large",
        chunk_tokens: int = 480,
        overlap_tokens: int = 80,
        min_chunk_tokens: int = 50,
    ) -> None:
        self.chunk_tokens = max(chunk_tokens, 64)
        self.overlap_tokens = max(min(overlap_tokens, self.chunk_tokens // 4), 0)
        self.min_chunk_tokens = max(min_chunk_tokens, 20)
        self.char_ratio = 4  # rough approximation characters per token
        self.tokenizer = None
        
        self.splitter = SemanticSplitter(char_ratio=self.char_ratio)
        
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(embedding_model)
            except Exception:
                self.tokenizer = None

    def build(
        self,
        file_id: str,
        text: str,
        page_mapping: Optional[List[tuple[int, int, int]]] = None,
        chunk_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        text_blocks: Optional[List["TextBlockInfo"]] = None,
    ) -> List[ChunkPayload]:
        """
        Build chunks from text with optional page mapping and spatial metadata.

        Args:
            file_id: Unique identifier for the source file
            text: Full text content to chunk
            page_mapping: Optional list of (start_offset, end_offset, page_num) tuples
            chunk_tokens: Override default chunk size in tokens
            overlap_tokens: Override default overlap in tokens
            text_blocks: Optional list of TextBlockInfo with spatial data for chunk area visualization
        """
        now = dt.datetime.now(dt.timezone.utc)

        if not text or not text.strip():
            return []

        # Use instance defaults if not provided
        target_chunk_tokens = chunk_tokens if chunk_tokens is not None else self.chunk_tokens
        target_overlap_tokens = overlap_tokens if overlap_tokens is not None else self.overlap_tokens

        # Calculate target sizes in characters
        max_chunk_chars = target_chunk_tokens * self.char_ratio
        min_chunk_chars = self.min_chunk_tokens * self.char_ratio
        overlap_chars = target_overlap_tokens * self.char_ratio

        # Step 1: Split by markdown sections first
        # We need to port split_sections to SemanticSplitter or keep it here if it orchestrates logic
        # Ideally it belongs in splitter.
        sections = self._split_sections_wrapper(text, min_chunk_chars)

        payloads: List[ChunkPayload] = []
        ordinal = 0

        for section_path, section_body, section_start_offset in sections:
            if not section_body.strip():
                continue

            # Step 2: Within each section, perform semantic chunking
            chunks = self.splitter.semantic_chunk(
                section_body,
                max_chars=max_chunk_chars,
                min_chars=min_chunk_chars,
                overlap_chars=overlap_chars,
            )

            for chunk_text, rel_start, rel_end in chunks:
                clean_text = chunk_text.strip()
                if not clean_text:
                    continue

                char_count = len(clean_text)
                token_count = self._token_count(clean_text)

                # Skip chunks that are too small (unless it's the only chunk)
                if token_count < self.min_chunk_tokens and len(chunks) > 1:
                    continue

                # Calculate absolute offsets
                abs_start = section_start_offset + rel_start
                abs_end = section_start_offset + rel_end

                chunk_id = self._chunk_id(file_id, ordinal, abs_start)
                snippet = clean_text[:1600]  # Match embed_max_chars for reranker fallback

                metadata = {
                    "section_path": section_path,
                    "start_char": abs_start,
                    "end_char": abs_end,
                    "list_density": self.splitter.list_density(clean_text),
                }

                # Add page number(s) if page_mapping is available
                if page_mapping:
                    pages = self._get_pages_for_range(abs_start, abs_end, page_mapping)
                    if pages:
                        metadata["page_numbers"] = pages
                        metadata["page_start"] = pages[0]
                        metadata["page_end"] = pages[-1]

                # Compute spatial metadata from text_blocks
                page_num, bbox, source_regions = self._get_spatial_metadata(
                    abs_start, abs_end, text_blocks, page_mapping
                )

                payloads.append(
                    ChunkPayload(
                        chunk_id=chunk_id,
                        file_id=file_id,
                        ordinal=ordinal,
                        text=clean_text,
                        snippet=snippet,
                        token_count=token_count,
                        char_count=char_count,
                        section_path=section_path,
                        metadata=metadata,
                        created_at=now,
                        page_num=page_num,
                        bbox=bbox,
                        source_regions=source_regions,
                    )
                )
                ordinal += 1

        # Post-process: merge consecutive small chunks to avoid fragmentation
        payloads = self._merge_small_chunks(
            payloads, min_chars=min_chunk_chars, max_chars=max_chunk_chars,
            page_mapping=page_mapping, text_blocks=text_blocks
        )

        return payloads

    def _get_spatial_metadata(
        self,
        start: int,
        end: int,
        text_blocks: Optional[List["TextBlockInfo"]],
        page_mapping: Optional[List[tuple[int, int, int]]],
    ) -> Tuple[Optional[int], Optional[tuple[float, float, float, float]], Optional[List[SourceRegion]]]:
        """
        Compute spatial metadata for a chunk based on text_blocks.

        Returns:
            Tuple of (page_num, bbox, source_regions):
            - page_num: Primary page number (1-indexed) or None
            - bbox: Primary bounding box (x0, y0, x1, y1) normalized 0-1 or None
            - source_regions: List of all source regions for multi-block chunks or None
        """
        if not text_blocks:
            # Fall back to page_mapping for page_num only
            if page_mapping:
                pages = self._get_pages_for_range(start, end, page_mapping)
                if pages:
                    return pages[0], None, None
            return None, None, None

        # Find all text blocks that overlap with this chunk's character range
        overlapping_blocks: List["TextBlockInfo"] = []
        for block in text_blocks:
            # Check if block overlaps with chunk range
            if block.char_start < end and block.char_end > start:
                overlapping_blocks.append(block)

        if not overlapping_blocks:
            # No overlapping blocks found, fall back to page_mapping
            if page_mapping:
                pages = self._get_pages_for_range(start, end, page_mapping)
                if pages:
                    return pages[0], None, None
            return None, None, None

        # Build source regions from overlapping blocks
        source_regions: List[SourceRegion] = []
        for block in overlapping_blocks:
            source_regions.append(SourceRegion(
                page_num=block.page_num,
                bbox=block.bbox,
                confidence=block.confidence,
            ))

        # Primary page and bbox: use the first (or largest) overlapping block
        primary_block = overlapping_blocks[0]
        page_num = primary_block.page_num
        bbox = primary_block.bbox

        # If multiple blocks, compute merged bounding box for same page
        if len(overlapping_blocks) > 1:
            same_page_blocks = [b for b in overlapping_blocks if b.page_num == page_num]
            if same_page_blocks:
                # Merge bounding boxes on the same page
                x0 = min(b.bbox[0] for b in same_page_blocks)
                y0 = min(b.bbox[1] for b in same_page_blocks)
                x1 = max(b.bbox[2] for b in same_page_blocks)
                y1 = max(b.bbox[3] for b in same_page_blocks)
                bbox = (x0, y0, x1, y1)

        return page_num, bbox, source_regions if len(source_regions) > 1 else None
    
    def _split_sections_wrapper(self, text: str, min_chars: int) -> List[tuple[Optional[str], str, int]]:
        """
        Wraps splitter.split_sections but handles the merging logic which currently resides in ChunkingPipeline
        or we can move the merge logic to splitter completely.
        For now, let's keep the merge logic here since it depends on min_chars which is config-dependent.
        Actually, let's duplicate the logic to SemanticSplitter.split_sections fully if not already done.
        
        Checking SemanticSplitter.split_sections: it currently returns raw_sections without merging.
        I will implement the merging logic here using the raw sections.
        """
        raw_sections = self.splitter.split_sections(text)
        
        # Merge small sections
        merged_sections: List[tuple[Optional[str], str, int]] = []
        pending_body = ""
        pending_path: Optional[str] = None
        pending_start = 0
        
        for section_path, section_body, section_start in raw_sections:
            if pending_body:
                combined_body = pending_body + "\n" + section_body
                if len(combined_body.strip()) >= min_chars:
                    merged_sections.append((pending_path, combined_body, pending_start))
                    pending_body = ""
                    pending_path = None
                else:
                    pending_body = combined_body
            else:
                if len(section_body.strip()) < min_chars:
                    pending_body = section_body
                    pending_path = section_path
                    pending_start = section_start
                else:
                    merged_sections.append((section_path, section_body, section_start))
        
        if pending_body:
            if merged_sections:
                last_path, last_body, last_start = merged_sections[-1]
                merged_sections[-1] = (last_path, last_body + "\n" + pending_body, last_start)
            else:
                merged_sections.append((pending_path, pending_body, pending_start))
        
        return merged_sections if merged_sections else [(None, text, 0)]

    def _token_count(self, text: str) -> int:
        if self.tokenizer is None:
            return max(len(text) // self.char_ratio, 1)
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return max(len(text) // self.char_ratio, 1)

    @staticmethod
    def _chunk_id(file_id: str, ordinal: int, start_offset: int) -> str:
        digest = xxhash.xxh64()
        digest.update(f"{file_id}:{ordinal}:{start_offset}")
        return digest.hexdigest()

    @staticmethod
    def _get_pages_for_range(start: int, end: int, page_mapping: List[tuple[int, int, int]]) -> List[int]:
        pages = set()
        for page_start, page_end, page_num in page_mapping:
            if start < page_end and end > page_start:
                pages.add(page_num)
        return sorted(pages)

    def _merge_small_chunks(
        self,
        payloads: List[ChunkPayload],
        min_chars: int,
        max_chars: int,
        page_mapping: Optional[List[tuple[int, int, int]]] = None,
        text_blocks: Optional[List["TextBlockInfo"]] = None,
    ) -> List[ChunkPayload]:
        if len(payloads) <= 1:
            return payloads

        merged: List[ChunkPayload] = []
        pending: Optional[ChunkPayload] = None

        for payload in payloads:
            if pending is None:
                if payload.char_count < min_chars:
                    pending = payload
                else:
                    merged.append(payload)
            else:
                combined_chars = pending.char_count + payload.char_count + 2
                should_merge = (combined_chars <= max_chars) or (payload.char_count < min_chars and combined_chars <= max_chars * 1.5)

                if should_merge:
                    combined_text = pending.text + "\n\n" + payload.text
                    combined_token_count = self._token_count(combined_text)

                    new_metadata = pending.metadata.copy()
                    new_metadata["start_char"] = pending.metadata.get("start_char", 0)
                    new_metadata["end_char"] = payload.metadata.get("end_char", len(combined_text))

                    pending_section = pending.section_path or ""
                    current_section = payload.section_path or ""
                    if pending_section and current_section and pending_section != current_section:
                        combined_section = f"{pending_section} | {current_section}"
                    else:
                        combined_section = pending_section or current_section

                    if page_mapping:
                        pages = self._get_pages_for_range(
                            new_metadata["start_char"],
                            new_metadata["end_char"],
                            page_mapping
                        )
                        if pages:
                            new_metadata["page_numbers"] = pages
                            new_metadata["page_start"] = pages[0]
                            new_metadata["page_end"] = pages[-1]

                    # Merge spatial metadata
                    merged_page_num, merged_bbox, merged_source_regions = self._merge_spatial_metadata(
                        pending, payload
                    )

                    merged_payload = ChunkPayload(
                        chunk_id=pending.chunk_id,
                        file_id=pending.file_id,
                        ordinal=pending.ordinal,
                        text=combined_text,
                        snippet=combined_text[:400],
                        token_count=combined_token_count,
                        char_count=len(combined_text),
                        section_path=combined_section,
                        metadata=new_metadata,
                        created_at=pending.created_at,
                        page_num=merged_page_num,
                        bbox=merged_bbox,
                        source_regions=merged_source_regions,
                    )

                    if merged_payload.char_count < min_chars:
                        pending = merged_payload
                    else:
                        merged.append(merged_payload)
                        pending = None
                else:
                    merged.append(pending)
                    if payload.char_count < min_chars:
                        pending = payload
                    else:
                        merged.append(payload)
                        pending = None

        if pending is not None:
            if merged and merged[-1].char_count + pending.char_count + 2 <= max_chars:
                last = merged.pop()
                combined_text = last.text + "\n\n" + pending.text
                combined_token_count = self._token_count(combined_text)

                new_metadata = last.metadata.copy()
                new_metadata["end_char"] = pending.metadata.get("end_char", last.metadata.get("end_char", 0) + len(pending.text))

                if page_mapping:
                    pages = self._get_pages_for_range(
                        new_metadata.get("start_char", 0),
                        new_metadata["end_char"],
                        page_mapping
                    )
                    if pages:
                        new_metadata["page_numbers"] = pages
                        new_metadata["page_start"] = pages[0]
                        new_metadata["page_end"] = pages[-1]

                # Merge spatial metadata
                merged_page_num, merged_bbox, merged_source_regions = self._merge_spatial_metadata(
                    last, pending
                )

                merged.append(ChunkPayload(
                    chunk_id=last.chunk_id,
                    file_id=last.file_id,
                    ordinal=last.ordinal,
                    text=combined_text,
                    snippet=combined_text[:400],
                    token_count=combined_token_count,
                    char_count=len(combined_text),
                    section_path=last.section_path,
                    metadata=new_metadata,
                    created_at=last.created_at,
                    page_num=merged_page_num,
                    bbox=merged_bbox,
                    source_regions=merged_source_regions,
                ))
            else:
                merged.append(pending)

        for i, payload in enumerate(merged):
            payload.ordinal = i

        return merged

    def _merge_spatial_metadata(
        self,
        chunk1: ChunkPayload,
        chunk2: ChunkPayload,
    ) -> Tuple[Optional[int], Optional[tuple[float, float, float, float]], Optional[List[SourceRegion]]]:
        """
        Merge spatial metadata from two chunks being combined.

        Returns:
            Tuple of (page_num, bbox, source_regions)
        """
        # Collect all source regions from both chunks
        all_regions: List[SourceRegion] = []

        # Add existing source_regions or create from page_num/bbox
        if chunk1.source_regions:
            all_regions.extend(chunk1.source_regions)
        elif chunk1.page_num is not None and chunk1.bbox is not None:
            all_regions.append(SourceRegion(
                page_num=chunk1.page_num,
                bbox=chunk1.bbox,
                confidence=None,
            ))

        if chunk2.source_regions:
            all_regions.extend(chunk2.source_regions)
        elif chunk2.page_num is not None and chunk2.bbox is not None:
            all_regions.append(SourceRegion(
                page_num=chunk2.page_num,
                bbox=chunk2.bbox,
                confidence=None,
            ))

        if not all_regions:
            # Use page_num from first chunk if available
            return chunk1.page_num or chunk2.page_num, None, None

        # Primary page and bbox: use the first region
        page_num = all_regions[0].page_num
        bbox = all_regions[0].bbox

        # If all on same page, merge bounding boxes
        if all(r.page_num == page_num for r in all_regions):
            x0 = min(r.bbox[0] for r in all_regions)
            y0 = min(r.bbox[1] for r in all_regions)
            x1 = max(r.bbox[2] for r in all_regions)
            y1 = max(r.bbox[3] for r in all_regions)
            bbox = (x0, y0, x1, y1)

        # Return source_regions only if there are multiple
        source_regions = all_regions if len(all_regions) > 1 else None

        return page_num, bbox, source_regions
