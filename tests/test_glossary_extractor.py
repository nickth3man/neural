"""Tests for document glossary / metadata extraction (deterministic + mocked LLM)."""

from __future__ import annotations

from unittest.mock import patch

import neural.glossary_extractor as ge
from neural.corpus import TranscriptDocument, TranscriptLine
from neural.glossary_extractor import extract_document_metadata
from neural.metadata_types import ContentType, EpisodeManifestEntry, EpisodeType
from neural.openrouter import OpenRouterError


def _doc(lines: list[tuple[int, str]], title: str = "Test Episode") -> TranscriptDocument:
    transcript_lines = tuple(
        TranscriptLine(
            timestamp=f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}",
            start_seconds=s,
            text=text,
        )
        for s, text in lines
    )
    return TranscriptDocument(
        episode_title=title,
        source_file="t.txt",
        source_path="/t/t.txt",
        lines=transcript_lines,
    )


def test_extract_skip_llm() -> None:
    document = _doc(
        [
            (0, "Welcome back to Gills Arena presented by Underdog."),
            (70, "We got the legend Gilbert Arenas here with us."),
            (200, "Here's what's cracking in the arena today. Lakers trade rumors."),
        ]
    )
    out = extract_document_metadata(document, skip_llm=True)
    assert out.used_llm is False
    assert out.raw_response == ""
    assert any("Gilbert" in n for n in out.metadata.episode_roster)
    assert out.intro_window.lines


def test_extract_no_api_key() -> None:
    document = _doc([(0, "Welcome back to Gills Arena.")])
    out = extract_document_metadata(document, api_key=None)
    assert out.used_llm is False


@patch("neural.glossary_extractor.complete_chat")
def test_extract_openrouter_error_falls_back(mock_chat) -> None:
    mock_chat.side_effect = OpenRouterError("down")
    document = _doc([(0, "Welcome back to Gills Arena.")])
    out = extract_document_metadata(document, api_key="k")
    assert out.used_llm is False


@patch("neural.glossary_extractor.complete_chat")
def test_extract_invalid_json_falls_back(mock_chat) -> None:
    mock_chat.return_value = "not json"
    document = _doc([(0, "Welcome back to Gills Arena.")])
    out = extract_document_metadata(document, api_key="k")
    assert out.used_llm is False
    assert out.raw_response == "not json"


@patch("neural.glossary_extractor.complete_chat")
def test_extract_llm_success_merges(mock_chat) -> None:
    mock_chat.return_value = (
        '{"episode_type": "recap", "host_names": ["Gilbert Arenas"], '
        '"guest_names": [], "holiday_theme": "", "topic": "Finals recap", '
        '"subtopic": "", "content_type": "main_discussion", '
        '"episode_roster": ["Gilbert Arenas", "Nick Young"]}'
    )
    document = _doc(
        [
            (0, "Welcome back to Gills Arena."),
            (200, "Here's what's cracking. Lakers news."),
        ]
    )
    manifest = EpisodeManifestEntry(
        title="T",
        url_slug="slug",
        date="2024-01-01",
        description="D",
        transcript_url="https://example.com/t",
        file_path="gil/x.txt",
    )
    out = extract_document_metadata(document, api_key="k", manifest_entry=manifest)
    assert out.used_llm is True
    assert out.metadata.episode_type == EpisodeType.RECAP
    assert out.metadata.topic.startswith("Finals")
    assert out.metadata.episode_date == "2024-01-01"
    assert out.metadata.episode_url == "https://example.com/t"


def test_parse_llm_response_markdown_fence() -> None:
    raw = '```\n{"k": "v"}\n```'
    assert ge._parse_llm_response(raw) == {"k": "v"}


def test_parse_episode_type_and_content_type_invalid() -> None:
    assert ge._parse_episode_type("bad") == EpisodeType.REACTION
    assert ge._parse_content_type("bad") == ContentType.MAIN_DISCUSSION
