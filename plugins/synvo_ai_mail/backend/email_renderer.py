"""Convert a markdown email body into a styled HTML email."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Inline-CSS email template.  All styles are inlined for maximum
# compatibility across email clients (Gmail, Outlook, Apple Mail, etc.).
_EMAIL_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background-color:#f4f4f7;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;-webkit-text-size-adjust:100%%;-ms-text-size-adjust:100%%;">
  <table role="presentation" width="100%%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f7;">
    <tr><td align="center" style="padding:24px 16px;">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%%;background-color:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <!-- Body -->
        <tr><td style="padding:32px 36px;color:#333333;font-size:15px;line-height:1.65;">
          %(body_html)s
        </td></tr>
        <!-- Footer -->
        <tr><td style="padding:16px 36px 24px;border-top:1px solid #eaeaec;">
          <p style="margin:0;font-size:12px;line-height:1.5;color:#9a9aa2;">
            Sent via <span style="color:#6b4fbb;">Local Cocoa</span>
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def render_email_html(body: str) -> str:
    """Return a complete HTML email document from a markdown body string.

    Falls back to a simple ``<pre>`` wrapped version if the markdown
    library is unavailable or conversion fails.
    """
    body_html = _markdown_to_html(body)
    return _EMAIL_TEMPLATE % {"body_html": body_html}


def _markdown_to_html(text: str) -> str:
    """Best-effort markdown → HTML fragment conversion."""
    try:
        import markdown

        return markdown.markdown(
            text,
            extensions=["tables", "nl2br", "sane_lists"],
            output_format="html",
        )
    except Exception:
        logger.debug("markdown conversion failed, falling back to plain text", exc_info=True)
        # Preserve newlines as <br> and escape basic HTML entities
        import html as _html

        escaped = _html.escape(text)
        return escaped.replace("\n", "<br>\n")
