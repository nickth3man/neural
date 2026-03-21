"""Tests for transcript chunking."""

from neural.chunking import ChunkingConfig, chunk_transcript
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
