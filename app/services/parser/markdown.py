from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from .base import BaseParser, ParsedContent

_md = MarkdownIt()


class MarkdownParser(BaseParser):
    extensions = {"md", "markdown", "mdx"}

    def parse(self, path: Path) -> ParsedContent:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = _md.render(raw)
        
        # Extract title from the first H1 header if present
        metadata = {"source": "markdown"}
        first_line = raw.strip().split("\n")[0]
        if first_line.startswith("# "):
            title = first_line[2:].strip()
            if title:
                metadata["title"] = title

        truncated = self._truncate(text)
        return ParsedContent(text=truncated, metadata=metadata)
