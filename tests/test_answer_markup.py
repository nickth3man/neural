"""Tests for answer HTML rendering and sanitization."""

from __future__ import annotations

import pytest

from neural.answer_markup import markdown_to_safe_html, plain_text_to_safe_html


def test_markdown_heading_and_list() -> None:
    html_out = markdown_to_safe_html("# Title\n\n- a\n- b\n")
    assert "<h1>Title</h1>" in html_out
    assert "<ul>" in html_out
    assert "a" in html_out and "b" in html_out


def test_markdown_fenced_code() -> None:
    html_out = markdown_to_safe_html("```\n<script>x</script>\n```")
    assert "<pre>" in html_out or "<code>" in html_out
    assert "<script>" not in html_out


def test_markdown_strips_script_tags() -> None:
    html_out = markdown_to_safe_html("Hello<script>alert(1)</script>")
    assert "<script>" not in html_out.lower()
    assert "Hello" in html_out


def test_markdown_javascript_href_stripped() -> None:
    html_out = markdown_to_safe_html("[x](javascript:alert(1))")
    assert "javascript:" not in html_out.lower()


def test_markdown_https_link_kept() -> None:
    html_out = markdown_to_safe_html("[ok](https://example.com/path)")
    assert 'href="https://example.com/path"' in html_out


def test_plain_text_escapes_angle_brackets() -> None:
    html_out = plain_text_to_safe_html("<script>x</script>")
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_plain_text_preserves_line_breaks() -> None:
    html_out = plain_text_to_safe_html("a\nb")
    assert "<br>" in html_out
    assert "a" in html_out and "b" in html_out


@pytest.mark.parametrize("fn", [markdown_to_safe_html, plain_text_to_safe_html])
def test_empty_yields_empty(fn) -> None:
    assert fn("") == ""
    assert fn("   \n  ") == ""
