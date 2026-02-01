from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, List

from markitdown import MarkItDown


@dataclass
class TextBlockInfo:
    """
    Represents a text block extracted from a document with spatial information.
    Used for chunk area visualization feature.
    """
    text: str
    page_num: int  # 1-indexed page number
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) normalized 0-1
    char_start: int  # Start character offset in the full text
    char_end: int  # End character offset in the full text
    confidence: Optional[float] = None  # OCR confidence if applicable
    block_type: str = "text"  # 'text', 'table', 'image', 'code'


@dataclass
class ParsedContent:
    text: str
    metadata: dict
    preview_image: Optional[bytes] = None
    duration_seconds: Optional[float] = None
    page_count: Optional[int] = None
    attachments: dict = field(default_factory=dict)
    # Maps character offset ranges to page numbers: [(start_offset, end_offset, page_num), ...]
    page_mapping: list[tuple[int, int, int]] = field(default_factory=list)
    # Text blocks with spatial information for chunk area visualization (optional, backward compatible)
    text_blocks: Optional[List[TextBlockInfo]] = None


class BaseParser(abc.ABC):
    extensions: set[str] = set()

    def __init__(self, *, max_chars: int = 4000) -> None:
        self.max_chars = max_chars
        self.md = MarkItDown()

    @abc.abstractmethod
    def parse(self, path: Path) -> ParsedContent:
        raise NotImplementedError

    def _truncate(self, text: str) -> str:
        if self.max_chars and len(text) > self.max_chars:
            return text[: self.max_chars]
        return text

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        return path.suffix.lower().lstrip(".") in cls.extensions


def select_parser(parsers: Iterable[BaseParser], path: Path) -> Optional[BaseParser]:
    for parser in parsers:
        if parser.can_handle(path):
            return parser
    return None
