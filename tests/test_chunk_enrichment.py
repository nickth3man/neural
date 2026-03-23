"""Tests for chunk-level enrichment (deterministic paths and mocked LLM)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import neural.chunk_enrichment as ce
from neural.chunk_enrichment import enrich_chunk, enrich_chunks
from neural.chunking import TranscriptChunk
from neural.metadata_types import (
    ClaimType,
    ContentType,
    ConversationType,
    DocumentMetadata,
    EpisodeType,
    SourceConfidence,
    Stance,
)
from neural.openrouter import OpenRouterError


def _sample_chunk(**overrides: object) -> TranscriptChunk:
    base = dict(
        episode_title="Ep",
        source_file="x.txt",
        start_timestamp="00:00:10",
        end_timestamp="00:00:20",
        start_seconds=10,
        end_seconds=20,
        chunk_text="Some main discussion text about the Lakers.",
        line_count=2,
    )
    base.update(overrides)
    return TranscriptChunk(**base)


def _doc_meta() -> DocumentMetadata:
    return DocumentMetadata(
        episode_type=EpisodeType.REACTION,
        host_names=("Gilbert Arenas",),
        guest_names=(),
        episode_roster=("Gilbert Arenas", "Nick Young"),
        topic="Lakers",
    )


def test_enrich_chunk_skip_llm() -> None:
    chunk = _sample_chunk()
    out = enrich_chunk(chunk, _doc_meta(), 0, skip_llm=True)
    assert out.used_llm is False
    assert out.raw_response == ""
    assert out.enriched_chunk.llm_meta.stance == Stance.DESCRIPTIVE
    assert out.enriched_chunk.llm_meta.source_confidence == SourceConfidence.LOW
    assert out.enriched_chunk.chunk_meta.chunk_index == 0


def test_enrich_chunk_no_api_key() -> None:
    chunk = _sample_chunk()
    out = enrich_chunk(chunk, _doc_meta(), 1, api_key=None, skip_llm=False)
    assert out.used_llm is False


@patch("neural.chunk_enrichment.complete_chat")
def test_enrich_chunk_openrouter_error_falls_back(mock_chat) -> None:
    mock_chat.side_effect = OpenRouterError("fail")
    chunk = _sample_chunk()
    out = enrich_chunk(chunk, _doc_meta(), 0, api_key="k")
    assert out.used_llm is False
    assert out.raw_response == ""


@patch("neural.chunk_enrichment.complete_chat")
def test_enrich_chunk_invalid_json_falls_back(mock_chat) -> None:
    mock_chat.return_value = "not json {{{"
    chunk = _sample_chunk()
    out = enrich_chunk(chunk, _doc_meta(), 0, api_key="k")
    assert out.used_llm is False
    assert "not json" in out.raw_response


@patch("neural.chunk_enrichment.complete_chat")
def test_enrich_chunk_llm_success(mock_chat) -> None:
    mock_chat.return_value = (
        '{"speaker": "Nick Young", "topic": "Lakers", "subtopic": "", '
        '"sentiment": "neutral", "stance": "critical", "source_confidence": "high", '
        '"speaker_evidence": "Nick said it", "topic_evidence": "Lakers mention here"}'
    )
    chunk = _sample_chunk()
    out = enrich_chunk(chunk, _doc_meta(), 0, api_key="k")
    assert out.used_llm is True
    assert out.enriched_chunk.llm_meta.speaker == "Nick Young"
    assert out.enriched_chunk.llm_meta.stance == Stance.CRITICAL
    assert out.enriched_chunk.llm_meta.source_confidence == SourceConfidence.HIGH
    assert out.enriched_chunk.llm_meta.speaker_anchor is not None


@patch("neural.chunk_enrichment.complete_chat")
def test_enrich_chunk_unknown_speaker_maps_to_roster_first(mock_chat) -> None:
    mock_chat.return_value = '{"speaker": "Not On Roster"}'
    chunk = _sample_chunk()
    out = enrich_chunk(chunk, _doc_meta(), 0, api_key="k")
    assert out.enriched_chunk.llm_meta.speaker == "Gilbert Arenas"


def test_enrich_chunks_iterates() -> None:
    chunks = [_sample_chunk(chunk_text="a"), _sample_chunk(chunk_text="b")]
    results = enrich_chunks(chunks, _doc_meta(), skip_llm=True)
    assert len(results) == 2
    assert results[0].enriched_chunk.chunk_text == "a"
    assert results[1].enriched_chunk.chunk_text == "b"


def test_classify_chunk_content_type_early_promo() -> None:
    ch = _sample_chunk(chunk_text="Use promo code WINNER on Underdog")
    meta = ce._classify_chunk_content_type(ch, 0)
    assert meta == ContentType.PROMO_READ


def test_classify_chunk_content_type_ad_read() -> None:
    ch = _sample_chunk(chunk_text="This segment is sponsored by our friends")
    meta = ce._classify_chunk_content_type(ch, 0)
    assert meta == ContentType.AD_READ


def test_classify_chunk_content_type_cast_intro() -> None:
    ch = _sample_chunk(chunk_text="Welcome back, here with us today is the crew")
    meta = ce._classify_chunk_content_type(ch, 2)
    assert meta == ContentType.CAST_INTRO


def test_classify_chunk_content_type_topic_rundown() -> None:
    ch = _sample_chunk(chunk_text="Here's what's cracking in the arena today")
    meta = ce._classify_chunk_content_type(ch, 3)
    assert meta == ContentType.TOPIC_RUNDOWN


def test_classify_chunk_content_type_outro() -> None:
    ch = _sample_chunk(chunk_text="Thanks for watching, see you next time")
    meta = ce._classify_chunk_content_type(ch, 10)
    assert meta == ContentType.OUTRO


def test_classify_chunk_content_type_cold_open() -> None:
    ch = _sample_chunk(chunk_text="Just kicking off the show today")
    assert ce._classify_chunk_content_type(ch, 0) == ContentType.COLD_OPEN_BANTER


def test_classify_chunk_content_type_main_discussion() -> None:
    ch = _sample_chunk(chunk_text="The pick and roll defense was solid")
    assert ce._classify_chunk_content_type(ch, 5) == ContentType.MAIN_DISCUSSION


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("They debate versus each other", ConversationType.DEBATE),
        ("Let me analyze the statistics here", ConversationType.ANALYSIS),
        ("Wow oh my shocking play", ConversationType.REACTION),
        ("Interview ask tell us about your career", ConversationType.INTERVIEW),
        ("News report announced today", ConversationType.NEWS_ROUNDUP),
        ("Download the app promo code sponsor", ConversationType.PROMOTION),
        ("Random couch talk", ConversationType.BANTER),
    ],
)
def test_classify_conversation_type(text: str, expected: ConversationType) -> None:
    assert ce._classify_conversation_type(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I predict they will be champions", ClaimType.PREDICTION),
        ("Rumor reportedly sources say", ClaimType.RUMOR),
        ("Story remember when back in the day", ClaimType.ANECDOTE),
        ("Fact actually statistics data", ClaimType.FACT),
        ("I think he is overrated", ClaimType.OPINION),
    ],
)
def test_classify_claim_type(text: str, expected: ClaimType) -> None:
    assert ce._classify_claim_type(text) == expected


def test_parse_llm_response_strips_fence() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert ce._parse_llm_response(raw) == {"a": 1}


def test_parse_llm_response_embedded_json() -> None:
    raw = 'Here: {"x": "y"} trailing'
    assert ce._parse_llm_response(raw) == {"x": "y"}


def test_parse_stance_and_confidence_invalid_defaults() -> None:
    assert ce._parse_stance("nope") == Stance.DESCRIPTIVE
    assert ce._parse_confidence("nope") == SourceConfidence.MEDIUM
