"""Tool executor – maps tool names to actual backend service calls."""
from __future__ import annotations

import json
import logging
from typing import Any

from core.context import get_search_engine, get_storage, get_service
from core.models import QaRequest

logger = logging.getLogger(__name__)


# ── Individual tool functions ───────────────────────────────────────────
# Each function receives the tool arguments dict and returns a string result.


async def execute_workspace_search(args: dict[str, Any]) -> str:
    """Search indexed documents, optionally filtered to a specific file."""
    query = args.get("query", "")
    limit = min(int(args.get("limit", 5)), 20)
    file_filter = args.get("file_filter", "").strip()

    storage = get_storage()
    engine = get_search_engine()

    # Resolve file_filter name -> file_ids
    file_ids: list[str] | None = None
    if file_filter:
        matched = storage.find_files_by_name(file_filter)
        if not matched:
            return json.dumps({"total": 0, "hits": [], "message": f"No file matching '{file_filter}' found."})
        file_ids = [f.id for f in matched]

    # Use FTS search with optional file_ids filter for speed (no LLM calls)
    raw_hits = storage.search_snippets_fts(query, limit=limit, file_ids=file_ids)

    hits_data = []
    for h in raw_hits[:limit]:
        hits_data.append({
            "file_id": h.file_id,
            "file": h.metadata.get("name", ""),
            "score": round(h.score, 3),
            "content": (h.snippet or "")[:1500],
            "chunk_id": h.chunk_id,
            "page": h.page_num,
        })
    return json.dumps({"total": len(raw_hits), "hits": hits_data}, ensure_ascii=False)


async def execute_workspace_qa(args: dict[str, Any]) -> str:
    """Answer a question using the full RAG pipeline (expensive, use as fallback)."""
    question = args.get("question", "")

    engine = get_search_engine()
    payload = QaRequest(query=question, mode="qa", search_mode="knowledge")
    result = await engine.answer(payload)

    sources = [h.metadata.get("name", h.file_id) for h in result.hits[:5]]
    return json.dumps({
        "answer": result.answer,
        "sources": sources,
    }, ensure_ascii=False)


async def execute_get_document_chunks(args: dict[str, Any]) -> str:
    """Retrieve text chunks from a specific document by file name or file_id."""
    file_id = args.get("file_id", "").strip()
    file_name = args.get("file_name", "").strip()
    query = args.get("query", "").strip()
    limit = min(int(args.get("limit", 5)), 20)

    storage = get_storage()

    # Resolve file
    if not file_id and file_name:
        matched = storage.find_files_by_name(file_name)
        if not matched:
            return json.dumps({"error": f"No file matching '{file_name}' found."})
        file_id = matched[0].id
    elif not file_id:
        return json.dumps({"error": "Provide either file_id or file_name."})

    if query:
        # Search within this file's chunks for relevant passages
        hits = storage.search_snippets_fts(query, limit=limit, file_ids=[file_id])
        chunks_data = []
        for h in hits:
            chunks_data.append({
                "chunk_id": h.chunk_id,
                "content": (h.snippet or "")[:2000],
                "score": round(h.score, 3),
                "page": h.page_num,
            })
        return json.dumps({"file_id": file_id, "total": len(hits), "chunks": chunks_data}, ensure_ascii=False)
    else:
        # Return first N chunks (ordered by ordinal)
        all_chunks = storage.chunks_for_file(file_id)
        chunks_data = []
        for c in all_chunks[:limit]:
            chunks_data.append({
                "chunk_id": c.chunk_id,
                "ordinal": c.ordinal,
                "content": (c.text or c.snippet or "")[:2000],
                "page": c.page_num,
                "section": c.section_path,
            })
        return json.dumps({
            "file_id": file_id,
            "total_chunks": len(all_chunks),
            "returned": len(chunks_data),
            "chunks": chunks_data,
        }, ensure_ascii=False)


async def execute_list_files(args: dict[str, Any]) -> str:
    """List indexed files."""
    limit = min(int(args.get("limit", 20)), 50)
    kind_filter = args.get("kind")

    storage = get_storage()
    files, total = storage.list_files(limit=limit)

    file_list = []
    for f in files:
        if kind_filter and f.kind != kind_filter:
            continue
        file_list.append({
            "name": f.name,
            "kind": f.kind,
            "size_bytes": f.size,
        })
    return json.dumps({"total": total, "files": file_list[:limit]}, ensure_ascii=False)


# ── Executor map ────────────────────────────────────────────────────────


async def execute_list_email_accounts(args: dict[str, Any]) -> str:
    """List connected email accounts."""
    try:
        email_service = get_service("synvo_ai_mail")
        if not email_service:
            raise RuntimeError("Email plugin is not installed/running.")
        accounts = email_service.list_accounts()
        return json.dumps([{
            "id": a.id,
            "label": a.label,
            "protocol": a.protocol,
            "username": a.username,
            "total_messages": a.total_messages,
        } for a in accounts], ensure_ascii=False)
    except ImportError:
        return json.dumps({"accounts": [], "message": "Email plugin is not installed."})
    except Exception as exc:
        return json.dumps({"error": f"Email service unavailable: {exc}"})


async def execute_search_emails(args: dict[str, Any]) -> str:
    """List recent emails from an account."""
    account_id = args.get("account_id", "")
    limit = min(int(args.get("limit", 20)), 50)
    try:
        email_service = get_service("synvo_ai_mail")
        if not email_service:
            raise RuntimeError("Email plugin is not installed/running.")
        messages = email_service.list_messages(account_id, limit)
        return json.dumps([{
            "id": m.id,
            "subject": m.subject,
            "sender": m.sender,
            "recipients": m.recipients,
            "sent_at": m.sent_at.isoformat() if m.sent_at else None,
            "preview": m.preview,
        } for m in messages], ensure_ascii=False)
    except ImportError:
        return json.dumps({"error": "Email plugin is not installed."})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


async def execute_send_email(args: dict[str, Any]) -> str:
    """Send an email via a connected account."""
    account_id = args.get("account_id", "")
    to_raw = args.get("to", "")
    subject = args.get("subject", "")
    body = args.get("body", "")

    # Parse comma-separated recipients
    to_list = [addr.strip() for addr in to_raw.split(",") if addr.strip()]
    if not to_list:
        return json.dumps({"error": "No recipient provided."})
    if not subject:
        return json.dumps({"error": "Subject is required."})

    try:
        from plugins.synvo_ai_mail.backend.models import EmailSendRequest
        email_service = get_service("synvo_ai_mail")
        if not email_service:
            raise RuntimeError("Email plugin is not installed/running.")
        request = EmailSendRequest(
            account_id=account_id,
            to=to_list,
            subject=subject,
            body=body,
        )
        result = await email_service.send_email(request)
        return json.dumps(result, ensure_ascii=False)
    except ImportError:
        return json.dumps({"error": "Email plugin is not installed."})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


TOOL_EXECUTORS: dict[str, Any] = {
    "workspace_search": execute_workspace_search,
    "get_document_chunks": execute_get_document_chunks,
    "workspace_qa": execute_workspace_qa,
    "list_files": execute_list_files,
    "list_email_accounts": execute_list_email_accounts,
    "search_emails": execute_search_emails,
    "send_email": execute_send_email,
}
