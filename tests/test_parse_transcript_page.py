"""Tests for parse_transcript_page."""

from scrape_podscripts import parse_transcript_page

TRANSCRIPT_HTML = """
<html><body>
<div class="podcast-transcript">
  <div class="single-sentence">Hello world.</div>
  <h2>Chapter One: Opening</h2>
  <div class="single-sentence">Second line.</div>
  <p class="noise">Ignored paragraph without single-sentence.</p>
</div>
</body></html>
"""


def test_parse_transcript_page_extracts_sentences_and_chapters() -> None:
    lines = parse_transcript_page(TRANSCRIPT_HTML)
    assert lines == [
        "Hello world.",
        "Chapter One: Opening",
        "Second line.",
    ]


def test_parse_transcript_page_empty_without_container() -> None:
    assert parse_transcript_page("<html><body><p>no transcript</p></body></html>") == []
