from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

from core.config import settings
from core.context import get_storage, get_indexer
from core.models import (
    FileRecord,
    FileListResponse,
    FolderRecord,
    ChunkSnapshot,
    PrivacyLevel,
)
from core.request_context import get_request_context
from core.vector_store import get_vector_store
from services.storage import IndexStorage


# Request/Response models for privacy endpoints
class PrivacyUpdateRequest(BaseModel):
    privacy_level: PrivacyLevel


class PrivacyUpdateResponse(BaseModel):
    file_id: str
    privacy_level: PrivacyLevel
    updated: bool

router = APIRouter(prefix="/files", tags=["files"])


def _resolve_page_number(meta: dict | None) -> int | None:
    if not meta:
        return None
    for key in ("page_number", "page", "page_start"):
        value = meta.get(key)
        if isinstance(value, int) and value > 0:
            return value
    page_numbers = meta.get("page_numbers")
    if isinstance(page_numbers, list) and page_numbers:
        first = page_numbers[0]
        if isinstance(first, int) and first > 0:
            return first
    return None


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "you",
    "your",
    "they",
    "their",
    "into",
    "over",
    "under",
    "between",
    "within",
    "also",
    "can",
    "may",
    "will",
    "would",
    "should",
    "could",
}


def _pick_highlight_terms(text: str, max_terms: int = 6) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]{4,}", text or "")
    seen: set[str] = set()
    picked: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        picked.append(token)
        if len(picked) >= max_terms:
            break
    return picked


@router.get("/chunks/{chunk_id}/highlight.png")
def get_chunk_highlight_png(chunk_id: str, zoom: float = Query(default=2.0, ge=1.0, le=4.0)) -> Response:
    """Render a PNG preview of the PDF page for this chunk, with a best-effort highlight.

    This is meant for UI preview (not for exporting), so highlighting is heuristic.
    """
    storage = get_storage()
    chunk = storage.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found.")

    record = storage.get_file(chunk.file_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    if (record.extension or "").lower() != "pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Highlight preview is only supported for PDFs.")

    pdf_path = Path(record.path)
    if not pdf_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF file not found on disk.")

    requested_page = _resolve_page_number(getattr(chunk, "metadata", None) or {})

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to open PDF: {exc}")

    try:
        terms = _pick_highlight_terms(getattr(chunk, "text", "") or "")

        # If page is unknown, try to locate the best matching page by searching terms.
        chosen_page_index: int | None = None
        rects: list[fitz.Rect] = []

        def collect_rects_for_page(p: fitz.Page) -> list[fitz.Rect]:
            out: list[fitz.Rect] = []
            for term in terms:
                try:
                    hits = p.search_for(term)
                except Exception:
                    hits = []
                for r in hits[:3]:
                    out.append(r)
            return out

        if requested_page is not None:
            chosen_page_index = max(0, min(doc.page_count - 1, requested_page - 1))
            page = doc.load_page(chosen_page_index)
            rects = collect_rects_for_page(page)
        else:
            # Search pages until we find at least one hit.
            for idx in range(doc.page_count):
                page = doc.load_page(idx)
                candidate = collect_rects_for_page(page)
                if candidate:
                    chosen_page_index = idx
                    rects = candidate
                    break
            if chosen_page_index is None:
                chosen_page_index = 0
                page = doc.load_page(0)

        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if rects:
            img_rgba = img.convert("RGBA")
            draw = ImageDraw.Draw(img_rgba, "RGBA")
            for r in rects:
                x0 = float(r.x0) * zoom
                y0 = float(r.y0) * zoom
                x1 = float(r.x1) * zoom
                y1 = float(r.y1) * zoom
                # Semi-transparent fill + subtle outline
                draw.rectangle([x0, y0, x1, y1], fill=(255, 230, 0, 45), outline=(255, 165, 0, 140), width=2)
            img = img_rgba

        from io import BytesIO

        out = BytesIO()
        img.save(out, format="PNG")
        return Response(content=out.getvalue(), media_type="image/png")
    finally:
        doc.close()


@router.get("/{file_id}/pages/{page_number}/image.png")
def get_pdf_page_image(
    file_id: str,
    page_number: int,
    zoom: float = Query(default=2.0, ge=0.5, le=8.0),
    bbox: str = Query(default=None, description="Bounding box to highlight, format: x0,y0,x1,y1 (normalized 0-1)"),
) -> Response:
    """Render a specific PDF page as a PNG image with optional bbox highlighting.

    This is used for the PDF preview component with region zoom capability.

    Args:
        file_id: The file ID
        page_number: Page number (1-indexed)
        zoom: Zoom level for rendering
        bbox: Optional bounding box to highlight, format: "x0,y0,x1,y1" (normalized 0-1 coordinates)
    """
    storage = get_storage()
    record = storage.get_file(file_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    if (record.extension or "").lower() != "pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="This endpoint only supports PDF files.")

    pdf_path = Path(record.path)
    if not pdf_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF file not found on disk.")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to open PDF: {exc}")

    try:
        # Validate page number (1-indexed from client, 0-indexed internally)
        page_index = page_number - 1
        if page_index < 0 or page_index >= doc.page_count:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Page {page_number} out of range. PDF has {doc.page_count} pages."
            )

        page = doc.load_page(page_index)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Parse and draw bounding box highlight if provided
        if bbox:
            try:
                parts = [float(p) for p in bbox.split(",")]
                if len(parts) == 4:
                    x0_norm, y0_norm, x1_norm, y1_norm = parts
                    page_width = page.rect.width
                    page_height = page.rect.height

                    # Convert normalized coords to pixel coords at zoom level
                    x0 = x0_norm * page_width * zoom
                    y0 = y0_norm * page_height * zoom
                    x1 = x1_norm * page_width * zoom
                    y1 = y1_norm * page_height * zoom

                    # Draw semi-transparent highlight
                    img_rgba = img.convert("RGBA")
                    draw = ImageDraw.Draw(img_rgba, "RGBA")
                    draw.rectangle(
                        [x0, y0, x1, y1],
                        fill=(255, 230, 0, 60),  # Yellow semi-transparent fill
                        outline=(255, 165, 0, 200),  # Orange outline
                        width=3
                    )
                    img = img_rgba
            except (ValueError, IndexError):
                pass  # Ignore invalid bbox format

        from io import BytesIO
        out = BytesIO()
        img.save(out, format="PNG")
        return Response(content=out.getvalue(), media_type="image/png")
    finally:
        doc.close()


@router.get("/chunks/{chunk_id}/region-preview.png")
def get_chunk_region_preview(
    chunk_id: str,
    zoom: float = Query(default=2.0, ge=1.0, le=4.0),
) -> Response:
    """Render a PNG preview of the PDF page for this chunk with its spatial region highlighted.

    Uses the chunk's stored bounding box metadata (if available) to highlight the exact source region.
    Falls back to text-based highlighting if no spatial metadata exists.
    """
    storage = get_storage()
    chunk = storage.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found.")

    record = storage.get_file(chunk.file_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    if (record.extension or "").lower() != "pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Region preview is only supported for PDFs.")

    pdf_path = Path(record.path)
    if not pdf_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF file not found on disk.")

    # Check if chunk has spatial metadata
    has_spatial_metadata = chunk.page_num is not None and chunk.bbox is not None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to open PDF: {exc}")

    try:
        if has_spatial_metadata:
            # Use spatial metadata for precise highlighting
            page_index = chunk.page_num - 1
            if page_index < 0 or page_index >= doc.page_count:
                page_index = 0

            page = doc.load_page(page_index)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Draw the bbox highlight
            page_width = page.rect.width
            page_height = page.rect.height
            x0_norm, y0_norm, x1_norm, y1_norm = chunk.bbox

            x0 = x0_norm * page_width * zoom
            y0 = y0_norm * page_height * zoom
            x1 = x1_norm * page_width * zoom
            y1 = y1_norm * page_height * zoom

            img_rgba = img.convert("RGBA")
            draw = ImageDraw.Draw(img_rgba, "RGBA")
            draw.rectangle(
                [x0, y0, x1, y1],
                fill=(255, 230, 0, 60),  # Yellow semi-transparent fill
                outline=(255, 165, 0, 200),  # Orange outline
                width=3
            )

            # If there are multiple source regions, draw them all
            if chunk.source_regions:
                for region in chunk.source_regions:
                    if region.page_num == chunk.page_num:
                        rx0, ry0, rx1, ry1 = region.bbox
                        draw.rectangle(
                            [rx0 * page_width * zoom, ry0 * page_height * zoom,
                             rx1 * page_width * zoom, ry1 * page_height * zoom],
                            fill=(200, 230, 255, 40),  # Light blue for additional regions
                            outline=(100, 150, 255, 150),
                            width=2
                        )
            img = img_rgba
        else:
            # Fall back to text-based highlighting (existing behavior)
            requested_page = _resolve_page_number(getattr(chunk, "metadata", None) or {})
            terms = _pick_highlight_terms(getattr(chunk, "text", "") or "")

            chosen_page_index: int | None = None
            rects: list[fitz.Rect] = []

            def collect_rects_for_page(p: fitz.Page) -> list[fitz.Rect]:
                out: list[fitz.Rect] = []
                for term in terms:
                    try:
                        hits = p.search_for(term)
                    except Exception:
                        hits = []
                    for r in hits[:3]:
                        out.append(r)
                return out

            if requested_page is not None:
                chosen_page_index = max(0, min(doc.page_count - 1, requested_page - 1))
                page = doc.load_page(chosen_page_index)
                rects = collect_rects_for_page(page)
            else:
                for idx in range(doc.page_count):
                    page = doc.load_page(idx)
                    candidate = collect_rects_for_page(page)
                    if candidate:
                        chosen_page_index = idx
                        rects = candidate
                        break
                if chosen_page_index is None:
                    chosen_page_index = 0
                    page = doc.load_page(0)

            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if rects:
                img_rgba = img.convert("RGBA")
                draw = ImageDraw.Draw(img_rgba, "RGBA")
                for r in rects:
                    x0 = float(r.x0) * zoom
                    y0 = float(r.y0) * zoom
                    x1 = float(r.x1) * zoom
                    y1 = float(r.y1) * zoom
                    draw.rectangle([x0, y0, x1, y1], fill=(255, 230, 0, 45), outline=(255, 165, 0, 140), width=2)
                img = img_rgba

        from io import BytesIO
        out = BytesIO()
        img.save(out, format="PNG")
        return Response(content=out.getvalue(), media_type="image/png")
    finally:
        doc.close()


@router.get("", response_model=FileListResponse)
def list_files(limit: int = Query(default=10000, ge=1, le=100000), offset: int = Query(default=0, ge=0), folder_id: str | None = None) -> FileListResponse:
    storage = get_storage()
    files, total = storage.list_files(limit=limit, offset=offset, folder_id=folder_id)
    return FileListResponse(files=files, total=total)


@router.get("/chunks/{chunk_id}", response_model=ChunkSnapshot)
def get_chunk(chunk_id: str) -> ChunkSnapshot:
    storage = get_storage()
    chunk = storage.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found.")
    return chunk


@router.get("/{file_id}/chunks", response_model=list[ChunkSnapshot])
def list_file_chunks(file_id: str) -> list[ChunkSnapshot]:
    """
    Return all chunks for a given file, ordered by their ordinal.

    This is used by the desktop UI to allow users to browse any chunk for a file.
    """
    storage = get_storage()
    record = storage.get_file(file_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return storage.chunks_for_file(file_id)


@router.get("/{file_id}", response_model=FileRecord)
def get_file(file_id: str) -> FileRecord:
    storage = get_storage()
    record = storage.get_file(file_id)

    # Backward compatibility: if not found by file_id, try as chunk_id
    if not record:
        record = storage.get_file_by_chunk_id(file_id)

    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return record


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_file(file_id: str) -> Response:
    storage = get_storage()
    indexer = get_indexer()
    record = storage.get_file(file_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    # Delete vectors
    try:
        get_vector_store().delete_by_filter(file_id=file_id)
    except Exception:
        # Fallback to chunk-based deletion if filter fails (e.g. old qdrant version?)
        # But delete_by_filter handles exceptions internally with logging.
        pass

    storage.delete_file(file_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ========================================
# Privacy Level Management Endpoints
# ========================================

@router.put("/{file_id}/privacy", response_model=PrivacyUpdateResponse)
def update_file_privacy(file_id: str, body: PrivacyUpdateRequest) -> PrivacyUpdateResponse:
    """
    Update the privacy level of a file.
    
    IMPORTANT: This endpoint can only be called from the local UI.
    External requests (API, MCP, plugins) cannot modify privacy settings.
    
    Privacy levels:
    - normal: File is accessible by all request sources
    - private: File is only accessible from local UI; blocked from external API, MCP, and plugins
    """
    ctx = get_request_context()
    
    # CRITICAL: Only local UI can modify privacy settings
    if ctx.source != "local_ui":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Privacy settings can only be changed from Local Cocoa UI"
        )
    
    storage = get_storage()
    # Use check_privacy=False to allow reading the file regardless of current privacy level
    record = storage.get_file(file_id, check_privacy=False)
    
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    
    # Update privacy level in database
    updated = storage.update_file_privacy(file_id, body.privacy_level)
    
    # Update privacy level in vector store
    get_vector_store().update_privacy_level(file_id, body.privacy_level)
    
    return PrivacyUpdateResponse(
        file_id=file_id,
        privacy_level=body.privacy_level,
        updated=updated,
    )


@router.get("/{file_id}/privacy")
def get_file_privacy(file_id: str) -> dict:
    """
    Get the privacy level of a file.
    
    Note: For external requests, this will return 404 for private files
    (to avoid leaking information about file existence).
    """
    storage = get_storage()
    record = storage.get_file(file_id)
    
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    
    return {
        "file_id": file_id,
        "privacy_level": record.privacy_level,
    }
