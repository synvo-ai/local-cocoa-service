"""Fixed tool definitions for agent mode v1.

Each tool is a ToolSpec describing what the agent can invoke.
The actual execution logic lives in executor.py.
"""
from __future__ import annotations

from .models import ToolParameter, ToolSpec

# ── Read-only tools ─────────────────────────────────────────────────────

WORKSPACE_SEARCH = ToolSpec(
    name="workspace_search",
    description=(
        "Search the user's indexed workspace for documents matching a query. "
        "Returns relevant chunks with content (up to 1500 chars each), scores, file names, and page numbers. "
        "Use this first when you need to find information from the user's files. "
        "You can optionally filter to a specific file by name."
    ),
    parameters=[
        ToolParameter(name="query", type="string", description="The search query"),
        ToolParameter(name="limit", type="integer", description="Max results to return (1-20, default 5)", required=False),
        ToolParameter(name="file_filter", type="string", description="Only search within files whose name contains this string", required=False),
    ],
)

WORKSPACE_QA = ToolSpec(
    name="workspace_qa",
    description=(
        "FALLBACK: Ask a complex question over the user's indexed documents using the full RAG pipeline. "
        "This is expensive (many internal LLM calls). Only use this when workspace_search results "
        "are insufficient and you cannot answer from the chunks alone. "
        "Prefer workspace_search + get_document_chunks for most questions."
    ),
    parameters=[
        ToolParameter(name="question", type="string", description="The question to answer from documents"),
    ],
)

GET_DOCUMENT_CHUNKS = ToolSpec(
    name="get_document_chunks",
    description=(
        "Retrieve text content from a specific document. You can get chunks by file_id or file_name "
        "(from a prior workspace_search result). Optionally pass a query to search within that "
        "document, or omit query to get the first N chunks in order. "
        "Use this to read deeper into a document after workspace_search identified it as relevant."
    ),
    parameters=[
        ToolParameter(name="file_id", type="string", description="The file ID (from workspace_search result)", required=False),
        ToolParameter(name="file_name", type="string", description="The file name to look up (alternative to file_id)", required=False),
        ToolParameter(name="query", type="string", description="Optional query to find specific passages within the file", required=False),
        ToolParameter(name="limit", type="integer", description="Max chunks to return (1-20, default 5)", required=False),
    ],
)

LIST_FILES = ToolSpec(
    name="list_files",
    description=(
        "List files in the user's indexed workspace. Returns file names, types, and sizes. "
        "Use this to understand what files are available."
    ),
    parameters=[
        ToolParameter(name="limit", type="integer", description="Max files to list (1-50)", required=False),
        ToolParameter(name="kind", type="string", description="Filter by file type", required=False,
                      enum=["document", "spreadsheet", "presentation", "image", "audio", "video"]),
    ],
)

LIST_NOTES = ToolSpec(
    name="list_notes",
    description=(
        "List all notes in the user's notes collection. Returns note titles, "
        "IDs, and short previews."
    ),
    parameters=[],
)

GET_NOTE = ToolSpec(
    name="get_note",
    description="Get the full content of a specific note by its ID.",
    parameters=[
        ToolParameter(name="note_id", type="string", description="The note ID to retrieve"),
    ],
)


# ── Side-effect tools ───────────────────────────────────────────────────

CREATE_NOTE = ToolSpec(
    name="create_note",
    description=(
        "Create a new note. Returns the created note summary with its ID. "
        "Use this when the user explicitly asks you to save or write down something."
    ),
    parameters=[
        ToolParameter(name="title", type="string", description="Title of the note"),
        ToolParameter(name="body", type="string", description="Markdown body content of the note"),
    ],
    side_effect=True,
)

LIST_EMAIL_ACCOUNTS = ToolSpec(
    name="list_email_accounts",
    description=(
        "List the user's connected email accounts. Returns account IDs, labels, "
        "protocols, and usernames. Use this to discover which accounts are available "
        "before sending or searching emails."
    ),
    parameters=[],
)

SEARCH_EMAILS = ToolSpec(
    name="search_emails",
    description=(
        "List recent emails from a specific email account. Returns subject, sender, "
        "recipients, and date for each message. Use list_email_accounts first to get the account_id."
    ),
    parameters=[
        ToolParameter(name="account_id", type="string", description="The email account ID"),
        ToolParameter(name="limit", type="integer", description="Max emails to return (1-50, default 20)", required=False),
    ],
)

SEND_EMAIL = ToolSpec(
    name="send_email",
    description=(
        "Send an email from one of the user's connected email accounts. "
        "Requires account_id (use list_email_accounts to find it), recipient address(es), "
        "subject, and body. Only use when the user explicitly asks to send an email."
    ),
    parameters=[
        ToolParameter(name="account_id", type="string", description="The email account ID to send from"),
        ToolParameter(name="to", type="string", description="Recipient email address(es), comma-separated if multiple"),
        ToolParameter(name="subject", type="string", description="Email subject line"),
        ToolParameter(name="body", type="string", description="Email body text"),
    ],
    side_effect=True,
)


# ── Master list ─────────────────────────────────────────────────────────

ALL_TOOLS: list[ToolSpec] = [
    WORKSPACE_SEARCH,
    GET_DOCUMENT_CHUNKS,
    LIST_FILES,
    LIST_NOTES,
    GET_NOTE,
    CREATE_NOTE,
    LIST_EMAIL_ACCOUNTS,
    SEARCH_EMAILS,
    SEND_EMAIL,
    WORKSPACE_QA,  # Fallback – expensive, listed last
]
