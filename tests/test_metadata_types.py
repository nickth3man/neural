"""Tests for neural.metadata_types — controlled vocabularies and dataclasses."""

from __future__ import annotations

import pytest
from neural.metadata_types import (
    ChunkAnchor,
    ChunkMetadata,
    ClaimType,
    ContentType,
    ConversationType,
    DocumentAnchor,
    DocumentMetadata,
    EnrichedChunk,
    EpisodeType,
    LlmMetadata,
    SourceConfidence,
    Stance,
    TopicCategory,
    chunk_enrichment_schema,
    document_extraction_schema,
)

# ============================================================================
# Enum Tests
# ============================================================================


class TestEpisodeType:
    def test_values(self) -> None:
        expected = {
            "season_premiere",
            "recap",
            "reaction",
            "prediction",
            "guest_interview",
            "trade_breakdown",
            "playoff_recap",
            "league_news",
        }
        assert {e.value for e in EpisodeType} == expected

    def test_from_string(self) -> None:
        assert EpisodeType("recap") == EpisodeType.RECAP

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            EpisodeType("invalid_type")


class TestContentType:
    def test_values(self) -> None:
        expected = {
            "ad_read",
            "cold_open_banter",
            "cast_intro",
            "topic_rundown",
            "main_discussion",
            "guest_interview",
            "promo_read",
            "outro",
        }
        assert {e.value for e in ContentType} == expected


class TestTopicCategory:
    def test_values(self) -> None:
        expected = {
            "team",
            "player",
            "league",
            "event",
            "business",
            "social_justice",
            "show_meta",
            "sponsor",
            "holiday",
        }
        assert {e.value for e in TopicCategory} == expected


class TestConversationType:
    def test_values(self) -> None:
        expected = {
            "debate",
            "analysis",
            "reaction",
            "banter",
            "interview",
            "news_roundup",
            "promotion",
        }
        assert {e.value for e in ConversationType} == expected


class TestClaimType:
    def test_values(self) -> None:
        expected = {"fact", "opinion", "prediction", "rumor", "anecdote", "promotion"}
        assert {e.value for e in ClaimType} == expected


class TestStance:
    def test_values(self) -> None:
        expected = {"supportive", "critical", "mixed", "skeptical", "descriptive"}
        assert {e.value for e in Stance} == expected


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestDocumentAnchor:
    def test_frozen(self) -> None:
        anchor = DocumentAnchor(line_index=0, timestamp="00:00:00", excerpt="test")
        with pytest.raises(AttributeError):
            anchor.line_index = 1  # type: ignore[misc]

    def test_to_dict(self) -> None:
        anchor = DocumentAnchor(line_index=5, timestamp="00:01:30", excerpt="hello world")
        # DocumentAnchor is frozen dataclass, asdict works
        from dataclasses import asdict

        d = asdict(anchor)
        assert d["line_index"] == 5
        assert d["timestamp"] == "00:01:30"
        assert d["excerpt"] == "hello world"


class TestChunkAnchor:
    def test_fields(self) -> None:
        anchor = ChunkAnchor(chunk_index=3, start_timestamp="00:05:00", excerpt="chunk text")
        assert anchor.chunk_index == 3
        assert anchor.start_timestamp == "00:05:00"
        assert anchor.excerpt == "chunk text"


class TestDocumentMetadata:
    def test_defaults(self) -> None:
        meta = DocumentMetadata()
        assert meta.show_title == "Gil's Arena"
        assert meta.episode_type == EpisodeType.REACTION
        assert meta.host_names == ()
        assert meta.guest_names == ()
        assert meta.episode_roster == ()

    def test_to_dict(self) -> None:
        meta = DocumentMetadata(
            episode_type=EpisodeType.GUEST_INTERVIEW,
            host_names=("Gilbert Arenas",),
            guest_names=("Dwight Howard",),
        )
        d = meta.to_dict()
        assert d["episode_type"] == "guest_interview"
        assert d["host_names"] == ("Gilbert Arenas",)
        assert d["guest_names"] == ("Dwight Howard",)


class TestChunkMetadata:
    def test_defaults(self) -> None:
        meta = ChunkMetadata()
        assert meta.chunk_index == 0
        assert meta.conversation_type == ConversationType.BANTER
        assert meta.claim_type == ClaimType.OPINION

    def test_to_dict(self) -> None:
        meta = ChunkMetadata(
            chunk_index=5,
            mentioned_people=("LeBron James",),
            mentioned_teams=("Lakers",),
        )
        d = meta.to_dict()
        assert d["chunk_index"] == 5
        assert d["mentioned_people"] == ("LeBron James",)


class TestLlmMetadata:
    def test_defaults(self) -> None:
        meta = LlmMetadata()
        assert meta.speaker == ""
        assert meta.stance == Stance.DESCRIPTIVE
        assert meta.source_confidence == SourceConfidence.MEDIUM


class TestEnrichedChunk:
    def test_defaults(self) -> None:
        chunk = EnrichedChunk(
            episode_title="Test Episode",
            source_file="test.txt",
            start_timestamp="00:00:00",
            end_timestamp="00:01:00",
            start_seconds=0,
            end_seconds=60,
            chunk_text="Hello world",
            line_count=2,
        )
        assert chunk.episode_title == "Test Episode"
        assert chunk.show_title == "Gil's Arena"
        assert chunk.episode_type == ""

    def test_to_dict(self) -> None:
        chunk = EnrichedChunk(
            episode_title="Test",
            source_file="test.txt",
            start_timestamp="00:00:00",
            end_timestamp="00:01:00",
            start_seconds=0,
            end_seconds=60,
            chunk_text="text",
            line_count=1,
        )
        d = chunk.to_dict()
        assert "chunk_meta" in d
        assert "llm_meta" in d


# ============================================================================
# JSON Schema Tests
# ============================================================================


class TestDocumentExtractionSchema:
    def test_schema_structure(self) -> None:
        schema = document_extraction_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "episode_type" in schema["properties"]
        assert "host_names" in schema["properties"]

    def test_episode_type_enum(self) -> None:
        schema = document_extraction_schema()
        ep_type = schema["properties"]["episode_type"]
        assert "enum" in ep_type
        assert "recap" in ep_type["enum"]
        assert "guest_interview" in ep_type["enum"]


class TestChunkEnrichmentSchema:
    def test_schema_structure(self) -> None:
        roster = ["Gilbert Arenas", "Nick Young"]
        schema = chunk_enrichment_schema(roster)
        assert schema["type"] == "object"
        assert "speaker" in schema["properties"]
        assert "conversation_type" in schema["properties"]

    def test_speaker_enum_constrained(self) -> None:
        roster = ["Gilbert Arenas", "Nick Young", "Brandon Jennings"]
        schema = chunk_enrichment_schema(roster)
        speaker = schema["properties"]["speaker"]
        assert "enum" in speaker
        assert set(speaker["enum"]) == set(roster)

    def test_empty_roster_fallback(self) -> None:
        schema = chunk_enrichment_schema([])
        speaker = schema["properties"]["speaker"]
        assert speaker["enum"] == ["unknown"]
