from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
import shutil

import fitz  # PyMuPDF

from .base import ParsedContent
from .pdf import PdfParser

logger = logging.getLogger(__name__)


class DocVisionParser(PdfParser):
    """Parse doc/docx by converting to PDF and reusing PdfParser."""

    extensions = {"doc", "docx"}

    def __init__(self) -> None:
        super().__init__()
        self.pdf_path: Path | None = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None

    def parse(self, path: Path) -> ParsedContent:
        self._cleanup_tempdir()
        pdf_path = self._convert_to_pdf(path)
        self.pdf_path = pdf_path
        page_texts = []
        if fitz:
            try:
                doc = fitz.open(str(self.pdf_path))

                if doc.is_encrypted:
                    doc.authenticate("")
                for index, page in enumerate(doc, start=1):

                    # # Use enhanced extraction with column detection
                    # page_text = self._text_parser._extract_text_enhanced(page)
                    # if not page_text:
                    page_text = ""
                    page_text = self._text_parser._extract_text_enhanced(page)
                    page_imgs_captions = self.image_caption(page)

                    if page_imgs_captions:
                        page_text = f"{page_text}\n{page_imgs_captions}"

                    page_texts.append(page_text)
                doc.close()
            except Exception as exc:
                logger.exception("Failed to parse PDF with DocVisionParser for path %s", path)
        final_text = "\n".join(page_texts)
        truncated = self._truncate(final_text)
        metadata = {}
        metadata = {
            "source": "doc/docx",
            "page_texts": final_text,
            "preview": self._truncate(truncated),
        }

        return ParsedContent(text=final_text, metadata=metadata)

    def _convert_to_pdf(self, path: Path) -> Path:
        soffice = shutil.which("soffice")
        if not soffice:
            raise FileNotFoundError("LibreOffice 'soffice' is required to convert doc/docx to PDF.")

        self._tempdir = tempfile.TemporaryDirectory(prefix="doc_vision_")
        tmp_dir = Path(self._tempdir.name)
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(tmp_dir), str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pdf_path = tmp_dir / f"{path.stem}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Failed to convert {path} to PDF.")
        return pdf_path

    def _cleanup_tempdir(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def close(self) -> None:
        self._cleanup_tempdir()

    def __del__(self) -> None:
        self._cleanup_tempdir()
