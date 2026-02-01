from __future__ import annotations

import logging
import io
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None
    HAS_NUMPY = False

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from .base import BaseParser, ParsedContent
from .pdf import PdfParser
from .vision_router import VisionRouter
from .img_2_wordbox import IMG2WORDS
from ..vlm.service import VisionProcessor

import asyncio


logger = logging.getLogger(__name__)


class PdfDeepParser(PdfParser):
    def __init__(self, threshold=0.85):
        self._text_parser = PdfParser()
        self._image_tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.ocr_workder = IMG2WORDS()
        self.vision_router = VisionRouter()
        self.threshold = threshold
        # Lazy initialization to avoid circular import with core.context
        self._vlm: VisionProcessor | None = None

    @property
    def vlm(self) -> VisionProcessor:
        """Lazily initialize VisionProcessor to avoid circular imports."""
        if self._vlm is None:
            from core.context import get_llm_client
            self._vlm = VisionProcessor(get_llm_client())
        return self._vlm

    @property
    def vlm(self) -> VisionProcessor:
        """Lazy initialization of VisionProcessor to avoid circular import."""
        if self._vlm is None:
            from core.context import get_llm_client
            vlm_client = get_llm_client()
            self._vlm = VisionProcessor(vlm_client)
        return self._vlm

    def turn_pdf_into_images(
        self,
        path: Path,
        *,
        as_array: bool = True,
        output_dir: Path | None = None,
        zoom: float = 2.0,
    ) -> dict:
        """Render PDF pages into images.

        Returns a dict of {page_number: <np.ndarray|path>}.
        
        Args:
            zoom: Rendering scale factor. 2.0 = 144 DPI (recommended for tables).
        """
        if not fitz:
            raise ImportError("PyMuPDF (fitz) is not installed.")
        if as_array and not HAS_NUMPY:
            raise ImportError("numpy is required for array output.")

        self._cleanup_image_tempdir()
        if not as_array and output_dir is None:
            self._image_tempdir = tempfile.TemporaryDirectory(prefix="pdf_pages_")
            output_dir = Path(self._image_tempdir.name)
            output_dir.mkdir(parents=True, exist_ok=True)
        elif output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        results: dict = {}
        doc = fitz.open(str(path))
        try:
            matrix = fitz.Matrix(zoom, zoom)

            for page_index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                if as_array:
                    data = np.frombuffer(pix.samples, dtype=np.uint8)
                    channels = pix.n
                    image = data.reshape((pix.height, pix.width, channels))
                    results[page_index] = image
                else:
                    assert output_dir is not None
                    image_path = output_dir / f"page_{page_index}.png"
                    pix.save(str(image_path))
                    results[page_index] = str(image_path)
        finally:
            doc.close()

        return results

    def _cleanup_image_tempdir(self) -> None:
        if self._image_tempdir is not None:
            self._image_tempdir.cleanup()
            self._image_tempdir = None

    def _array_to_bytes(self, image_array) -> bytes:
        """Convert a numpy image array to PNG bytes."""
        if Image is None:
            raise ImportError("Pillow is required to encode image arrays.")
        if not HAS_NUMPY:
            raise ImportError("numpy is required to encode image arrays.")
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        if image_array.ndim == 2:
            mode = "L"
        else:
            mode = "RGB"
            if image_array.shape[2] == 4:
                mode = "RGBA"
        with Image.fromarray(image_array, mode=mode) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()

    def parse(self, path: Path) -> ParsedContent:
        """
        Parse PDF by converting each page to an image.

        Returns ParsedContent with:
        - text: Empty or placeholder (VLM will generate the actual text)
        - attachments: Dict of page images {f"page_{num}": bytes}
        - page_count: Number of pages
        - page_mapping: Empty list (will be populated per-page after VLM processing)
        """
        try:
            # Use centralized vision processor to render images
            # Use zoom=2.0 for higher resolution (144 DPI instead of 72 DPI)
            # This improves VLM accuracy for tables and small text in financial reports
            attachments = self.turn_pdf_into_images(path, zoom=2.0)
        except Exception as e:
            raise ImportError(
                f"Failed to convert PDF to images: {e}. "
                "Ensure 'pymupdf' is installed (pip install pymupdf)."
            )
        page_count = len(attachments)

        # Use PdfParser's enhanced text extraction (with column detection)
        # This provides better quality text for fallback and hybrid mode
        page_texts = []
        page_mapping = []
        if fitz:
            try:
                doc = fitz.open(str(path))
                cursor = 0
                if doc.is_encrypted:
                    doc.authenticate("")
                for index, page in enumerate(doc, start=1):

                    # # Use enhanced extraction with column detection
                    # page_text = self._text_parser._extract_text_enhanced(page)
                    # if not page_text:
                    page_text = ""
                    page_img = attachments[index]

                    page_bboxs = self.ocr_workder.run(page_img)
                    page_text = self._text_parser._extract_text_from_bboxes(page_img, page_bboxs)
                    page_imgs_captions = self.image_caption(page)

                    if page_imgs_captions:
                        page_text = f"{page_text}\n{page_imgs_captions}"

                    page_bboxs_pure = [data[:4] for data in page_bboxs]
                    need_vlm = self.vision_router.run(page_img, page_bboxs_pure)
                    if need_vlm["bbox_ratio_effective"] <= self.threshold:
                        page_bytes = self._array_to_bytes(page_img)
                        caption = asyncio.run(self.vlm.process_image(page_bytes))
                        page_text = f"{page_text} caption:({caption})"

                    start = cursor
                    cursor = start + len(page_text)

                    if index < page_count:
                        cursor += 2  # "\n\n"

                    end = cursor

                    page_texts.append(page_text)
                    page_mapping.append((start, end, index))

                doc.close()
            except Exception:
                pass

        page_text_final = "\n\n".join([f"{page_content}" for index, page_content in enumerate(page_texts, start=1)])

        metadata = {
            "source": "pdf_vision",
            "pages": page_count,
            "processing_mode": "vision",
            "page_texts": page_texts,
        }
        attachments_metadata = {f"page_{num}": self._array_to_bytes(img) for num, img in attachments.items()}

        return ParsedContent(
            text=page_text_final,
            metadata=metadata,
            page_count=page_count,
            attachments=attachments_metadata,
            page_mapping=page_mapping,
        )

    def image_caption(self, page):
        images_to_be_caption = self._extract_images(page)
        if not images_to_be_caption:
            return ""

        async def _image_caption_async(images_to_be_caption):
            caption_tasks = [self.vlm._describe_image(img["image_bytes"]) for img in images_to_be_caption]
            results = await asyncio.gather(*caption_tasks)
            results_refine = [(images_to_be_caption[index]["bbox"], results[index]) for index in range(0, len(results))]

            return results_refine
        try:

            img_caption_results = asyncio.run(_image_caption_async(images_to_be_caption))
        except Exception as e:
            img_caption_results = []
        sorted_img_captions = self.sort_images(img_caption_results)
        image_caption_str = "\n".join([f"image{str(position)}:\ncaption:{caption}\n" for position, caption in sorted_img_captions])

        final_caption_str = f"images_caption result: {image_caption_str}"

        return final_caption_str

    def sort_images(self, bbox_caption_pairs, y_tol: float = 8.0) -> list[tuple]:
        """Sort captions by bbox reading order (top-to-bottom, left-to-right)."""
        if not bbox_caption_pairs:
            return []

        valid = []
        invalid = []
        for bbox, caption in bbox_caption_pairs:
            if not bbox or len(bbox) < 4:
                invalid.append((None, caption))
            else:
                valid.append((bbox, caption))

        if not valid:
            return invalid

        valid.sort(key=lambda item: (item[0][1], item[0][0]))
        lines = []
        for bbox, caption in valid:
            if not lines:
                lines.append([(bbox, caption)])
                continue
            line_y = lines[-1][0][0][1]
            if abs(bbox[1] - line_y) <= y_tol:
                lines[-1].append((bbox, caption))
            else:
                lines.append([(bbox, caption)])

        ordered = []
        for line in lines:
            line.sort(key=lambda item: item[0][0])
            ordered.extend([(bbox, caption) for bbox, caption in line])

        ordered.extend(invalid)
        return ordered
