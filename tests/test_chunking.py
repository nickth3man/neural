"""Tests for transcript chunking."""

import pytest

from neural.chunking import ChunkingConfig, TranscriptChunk, chunk_corpus, chunk_transcript
from neural.corpus import TranscriptDocument, TranscriptLine


def test_chunk_transcript_preserves_overlap_and_timestamps() -> None:
    document = TranscriptDocument(
        episode_title="test_episode",
        source_file="test_episode.txt",
        source_path="test_episode.txt",
        lines=tuple(
            TranscriptLine(
                timestamp=f"00:00:0{i}",
                start_seconds=i,
                text=f"Line {i} text",
            )
            for i in range(1, 7)
        ),
    )

    chunks = chunk_transcript(
        document,
        ChunkingConfig(lines_per_chunk=3, line_overlap=1, min_chunk_characters=1),
    )

    assert len(chunks) == 3
    assert chunks[0].start_timestamp == "00:00:01"
    assert chunks[0].end_timestamp == "00:00:03"
    assert chunks[1].start_timestamp == "00:00:03"
    assert chunks[1].end_timestamp == "00:00:05"
    assert chunks[2].start_timestamp == "00:00:05"
    assert chunks[2].end_timestamp == "00:00:06"
    assert chunks[1].chunk_text == "Line 3 text Line 4 text Line 5 text"


def test_chunk_transcript_empty_document() -> None:
    document = TranscriptDocument(
        episode_title="e",
        source_file="e.txt",
        source_path="e.txt",
        lines=(),
    )
    assert chunk_transcript(document) == []


def test_chunk_corpus_concatenates() -> None:
    doc_a = TranscriptDocument(
        episode_title="a",
        source_file="a.txt",
        source_path="a.txt",
        lines=tuple(
            TranscriptLine(timestamp=f"00:00:0{i}", start_seconds=i, text=f"L{i}")
            for i in range(1, 5)
        ),
    )
    doc_b = TranscriptDocument(
        episode_title="b",
        source_file="b.txt",
        source_path="b.txt",
        lines=tuple(
            TranscriptLine(timestamp=f"00:00:{i:02d}", start_seconds=i, text=f"X{i}")
            for i in range(10, 14)
        ),
    )
    cfg = ChunkingConfig(lines_per_chunk=2, line_overlap=0, min_chunk_characters=1)
    chunks = chunk_corpus([doc_a, doc_b], cfg)
    assert len(chunks) == 4
    assert chunks[0].episode_title == "a"
    assert chunks[-1].episode_title == "b"


def test_chunk_transcript_skips_tiny_trailing_window() -> None:
    document = TranscriptDocument(
        episode_title="e",
        source_file="e.txt",
        source_path="e.txt",
        lines=(
            TranscriptLine(timestamp="00:00:01", start_seconds=1, text="A " * 30),
            TranscriptLine(timestamp="00:00:02", start_seconds=2, text="B " * 30),
            TranscriptLine(timestamp="00:00:03", start_seconds=3, text="x"),
        ),
    )
    cfg = ChunkingConfig(lines_per_chunk=2, line_overlap=0, min_chunk_characters=10)
    chunks = chunk_transcript(document, cfg)
    assert len(chunks) == 1


def test_transcript_chunk_to_dict() -> None:
    chunk = TranscriptChunk(
        episode_title="e",
        source_file="f.txt",
        start_timestamp="00:00:01",
        end_timestamp="00:00:02",
        start_seconds=1,
        end_seconds=2,
        chunk_text="hi",
        line_count=1,
    )
    d = chunk.to_dict()
    assert d["chunk_text"] == "hi"
    assert d["line_count"] == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lines_per_chunk": 0}, "lines_per_chunk"),
        ({"line_overlap": -1}, "line_overlap"),
        ({"lines_per_chunk": 3, "line_overlap": 3}, "line_overlap"),
        ({"min_chunk_characters": 0}, "min_chunk_characters"),
    ],
)
def test_chunking_config_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ChunkingConfig(**kwargs)
