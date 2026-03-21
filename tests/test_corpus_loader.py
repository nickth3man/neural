"""Tests for transcript corpus loading."""

from pathlib import Path

import pytest
from neural.corpus import (
    format_timestamp,
    load_corpus,
    load_transcript,
    parse_timestamp,
    parse_transcript_line,
)


def test_parse_timestamp_and_format_timestamp_round_trip() -> None:
    assert parse_timestamp("00:01:11") == 71
    assert format_timestamp(71) == "00:01:11"


def test_parse_timestamp_edge_cases_tc001() -> None:
    assert parse_timestamp("00:00:00") == 0
    assert format_timestamp(0) == "00:00:00"
    assert parse_timestamp("23:59:59") == 23 * 3600 + 59 * 60 + 59
    assert format_timestamp(23 * 3600 + 59 * 60 + 59) == "23:59:59"


def test_parse_transcript_line_extracts_timestamp_and_text() -> None:
    parsed = parse_transcript_line("Starting point is 00:00:24Hello   world")
    assert parsed is not None
    assert parsed.timestamp == "00:00:24"
    assert parsed.start_seconds == 24
    assert parsed.text == "Hello world"


def test_load_transcript_merges_continuations_and_skips_leading_untimed_lines(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample_episode.txt"
    transcript_path.write_text(
        "\n".join(
            [
                "Preface without a timestamp",
                "Starting point is 00:00:10First sentence.",
                "Follow-up untimed continuation.",
                "Starting point is 00:00:20Second sentence.",
            ]
        ),
        encoding="utf-8",
    )

    document = load_transcript(transcript_path)

    assert document.episode_title == "sample_episode"
    assert document.source_file == "sample_episode.txt"
    assert document.skipped_prefix_lines == 1
    assert len(document.lines) == 2
    assert document.lines[0].text == "First sentence. Follow-up untimed continuation."
    assert document.lines[1].timestamp == "00:00:20"


def test_load_corpus_raises_when_directory_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="Transcript directory not found"):
        load_corpus(missing)


def test_load_corpus_raises_when_no_txt_files(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No transcript files found"):
        load_corpus(empty_dir)
