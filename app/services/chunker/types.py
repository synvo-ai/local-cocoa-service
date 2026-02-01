from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SourceRegion:
    """
    Represents a spatial region in the source document where chunk text originated.
    Coordinates are normalized (0-1) relative to page dimensions for resolution independence.
    """
    page_num: int  # 1-indexed page number
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) normalized 0-1
    confidence: Optional[float] = None  # OCR confidence if applicable

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceRegion":
        bbox = data.get("bbox", [0, 0, 1, 1])
        if isinstance(bbox, list):
            bbox = tuple(bbox)
        return cls(
            page_num=data.get("page_num", 1),
            bbox=bbox,
            confidence=data.get("confidence"),
        )


@dataclass
class TextBlock:
    """
    Represents a text block extracted from a document with spatial information.
    Used during parsing before chunking.
    """
    text: str
    page_num: int  # 1-indexed page number
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) normalized 0-1
    confidence: Optional[float] = None  # OCR confidence if applicable
    block_type: str = "text"  # 'text', 'table', 'image', 'code'

    def to_source_region(self) -> SourceRegion:
        return SourceRegion(
            page_num=self.page_num,
            bbox=self.bbox,
            confidence=self.confidence,
        )


@dataclass
class ChunkPayload:
    chunk_id: str
    file_id: str
    ordinal: int
    text: str
    snippet: str
    token_count: int
    char_count: int
    section_path: Optional[str]
    metadata: dict
    created_at: dt.datetime
    # Spatial metadata for chunk area visualization (optional, backward compatible)
    page_num: Optional[int] = None  # Primary page number (1-indexed)
    bbox: Optional[tuple[float, float, float, float]] = None  # Primary bbox (normalized 0-1)
    source_regions: Optional[List[SourceRegion]] = field(default=None)  # All source regions for multi-page/multi-block chunks


@dataclass
class SemanticBlock:
    """Represents a semantic unit (paragraph, list, code block, table, etc.)"""
    text: str
    block_type: str  # 'paragraph', 'list', 'code', 'table', 'heading'
    start_offset: int
    end_offset: int
    # Spatial metadata (optional, backward compatible)
    page_num: Optional[int] = None  # 1-indexed page number
    bbox: Optional[tuple[float, float, float, float]] = None  # (x0, y0, x1, y1) normalized 0-1
