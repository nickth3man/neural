"""Convert model answers to sanitized HTML (Markdown subset + plain text)."""

from __future__ import annotations

import html
from typing import Final

import markdown
import nh3

# Tags allowed after Markdown conversion.
_ALLOWED_TAGS: Final[set[str]] = {
    "p",
    "br",
    "strong",
    "em",
    "code",
    "pre",
    "ul",
    "ol",
    "li",
    "a",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
}

_ALLOWED_ATTRIBUTES: Final[dict[str, set[str]]] = {
    "a": {"href", "title"},
}

_ALLOWED_URL_SCHEMES: Final[set[str]] = {"http", "https", "mailto"}

_MARKDOWN_EXTENSIONS: Final[tuple[str, ...]] = (
    "fenced_code",
    "nl2br",
    "sane_lists",
)


def _sanitize_fragment(html_fragment: str) -> str:
    return nh3.clean(
        html_fragment,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        url_schemes=_ALLOWED_URL_SCHEMES,
    )


def markdown_to_safe_html(text: str) -> str:
    """Render Markdown to HTML, then sanitize with a strict allowlist.

    Suitable for LLM-generated answers. Empty or whitespace-only input yields "".
    """
    if not text.strip():
        return ""
    raw = markdown.markdown(
        text,
        extensions=list(_MARKDOWN_EXTENSIONS),
        output_format="html",
    )
    return _sanitize_fragment(raw).strip()


def plain_text_to_safe_html(text: str) -> str:
    """Escape plain text and wrap as a single safe paragraph (line breaks as <br>).

    Use for retrieval-only summaries and other non-Markdown prose so underscores
    and other characters are not interpreted as Markdown.
    """
    if not text.strip():
        return ""
    escaped = html.escape(text, quote=False)
    with_breaks = escaped.replace("\n", "<br>\n")
    fragment = f"<p>{with_breaks}</p>" if with_breaks.strip() else ""
    if not fragment:
        return ""
    return _sanitize_fragment(fragment).strip()
