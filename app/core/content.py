from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from core.config import settings
from services.parser import (
    AudioParser,
    BaseParser,
    DocParser,
    DocxParser,
    ImageParser,
    MarkdownParser,
    ParsedContent,
    PdfParser,
    PdfVisionParser,
    TextParser,
    VideoParser,
    GeneralParser,
    select_parser,
    PdfDeepParser
)

logger = logging.getLogger(__name__)


class ContentRouter:
    def __init__(self, parsers: Optional[Iterable[BaseParser]] = None) -> None:
        self._parsers_override = list(parsers) if parsers else None
        self._parsers: list[BaseParser] | None = None
        # Lazy initialization for parsers to avoid circular imports
        self._pdf_text_parser: PdfParser | None = None
        self._pdf_vision_parser: PdfDeepParser | None = None

    @property
    def parsers(self) -> list[BaseParser]:
        if self._parsers is None:
            self._parsers = self._parsers_override if self._parsers_override else self._default_parsers()
        return self._parsers

    @property
    def pdf_text_parser(self) -> PdfParser:
        if self._pdf_text_parser is None:
            self._pdf_text_parser = PdfParser()
        return self._pdf_text_parser

    @property
    def pdf_vision_parser(self) -> PdfDeepParser:
        if self._pdf_vision_parser is None:
            self._pdf_vision_parser = PdfDeepParser()
        return self._pdf_vision_parser

    def parse(self, path: Path, indexing_mode: str = "fast", **kwargs) -> ParsedContent:
        # Special handling for PDF to choose parser based on mode
        if path.suffix.lower() == ".pdf":
            logger.info(
                "📄 PDF parse: file=%s, indexing_mode=%s, pdf_mode=%s",
                path.name, indexing_mode, settings.pdf_mode
            )
            # FAST MODE: Always use PdfParser (PyMuPDF text extraction only)
            # - No OCR, no VLM
            # - Scanned PDFs will have empty text - they require Deep index for OCR
            if indexing_mode == "fast":
                logger.info("📄 Using PdfParser (fast text extraction, no VLM)")
                try:
                    result = self.pdf_text_parser.parse(path, **kwargs)
                    logger.info("📄 PdfParser result: %d pages, %d chars", result.page_count, len(result.text))
                    return result
                except TypeError:
                    result = self.pdf_text_parser.parse(path)
                    logger.info("📄 PdfParser result: %d pages, %d chars", result.page_count, len(result.text))
                    return result

            # DEEP MODE: Use PdfDeepParser (OCR + VLM)
            logger.info("📄 Using PdfDeepParser (vision/OCR mode)")
            try:
                return self.pdf_vision_parser.parse(path, **kwargs)
            except TypeError:
                return self.pdf_vision_parser.parse(path)

        parser = select_parser(self.parsers, path)
        if not parser:
            # raise ValueError(f"Unsupported file type for {path}")
            parser = GeneralParser()

        try:
            return parser.parse(path, **kwargs)
        except TypeError:
            return parser.parse(path)

    @staticmethod
    def _default_parsers() -> list[BaseParser]:
        # Choose PDF parser based on configuration
        pdf_parser = PdfDeepParser() if settings.pdf_mode == "vision" else PdfParser()

        return [
            TextParser(),
            MarkdownParser(),
            pdf_parser,  # Dynamic selection based on settings
            DocParser(),
            DocxParser(),
            ImageParser(),
            VideoParser(),
            AudioParser(),
            GeneralParser(),
        ]


content_router = ContentRouter()
