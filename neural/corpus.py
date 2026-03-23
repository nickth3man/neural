"""Transcript corpus loading utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TIMESTAMP_LINE_RE = re.compile(r"^Starting point is (?P<timestamp>\d{2}:\d{2}:\d{2})(?P<text>.*)$")


@dataclass(frozen=True)
class TranscriptLine:
    """One timestamped transcript line."""

    timestamp: str
    start_seconds: int
    text: str


@dataclass(frozen=True)
class TranscriptDocument:
    """Structured transcript document with provenance."""

    episode_title: str
    source_file: str
    source_path: str
    lines: tuple[TranscriptLine, ...]
    skipped_prefix_lines: int = 0


def parse_timestamp(timestamp: str) -> int:
    """Convert an ``HH:MM:SS`` string into total seconds."""
    parts = timestamp.split(":")
    if len(parts) != 3:
        msg = f"Invalid timestamp: {timestamp!r}"
        raise ValueError(msg)

    hours, minutes, seconds = (int(part) for part in parts)
    if minutes >= 60 or seconds >= 60 or min(hours, minutes, seconds) < 0:
        msg = f"Invalid timestamp: {timestamp!r}"
        raise ValueError(msg)
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(total_seconds: int) -> str:
    """Convert total seconds into canonical ``HH:MM:SS`` form."""
    if total_seconds < 0:
        msg = f"Timestamp cannot be negative: {total_seconds}"
        raise ValueError(msg)

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_text(text: str) -> str:
    """Collapse internal whitespace and trim surrounding space."""
    return " ".join(text.split())


def parse_transcript_line(raw_line: str) -> TranscriptLine | None:
    """Parse one raw transcript line into structured form."""
    stripped = raw_line.strip()
    if not stripped:
        return None

    match = TIMESTAMP_LINE_RE.match(stripped)
    if not match:
        return None

    timestamp = match.group("timestamp")
    text = normalize_text(match.group("text"))
    return TranscriptLine(
        timestamp=timestamp,
        start_seconds=parse_timestamp(timestamp),
        text=text,
    )


def load_transcript(path: Path) -> TranscriptDocument:
    """Load one transcript file into a structured document."""
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    parsed_lines: list[TranscriptLine] = []
    skipped_prefix_lines = 0

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        parsed = parse_transcript_line(stripped)
        if parsed is not None:
            parsed_lines.append(parsed)
            continue

        continuation_text = normalize_text(stripped)
        if parsed_lines:
            previous = parsed_lines[-1]
            parsed_lines[-1] = TranscriptLine(
                timestamp=previous.timestamp,
                start_seconds=previous.start_seconds,
                text=normalize_text(f"{previous.text} {continuation_text}"),
            )
        else:
            skipped_prefix_lines += 1

    return TranscriptDocument(
        episode_title=path.stem,
        source_file=path.name,
        source_path=str(path),
        lines=tuple(parsed_lines),
        skipped_prefix_lines=skipped_prefix_lines,
    )


def load_corpus(transcripts_dir: Path, limit: int | None = None) -> list[TranscriptDocument]:
    """Load all transcript files from a directory."""
    if not transcripts_dir.exists():
        msg = f"Transcript directory not found: {transcripts_dir}"
        raise FileNotFoundError(msg)

    transcript_paths = sorted(path for path in transcripts_dir.glob("*.txt") if path.is_file())
    if limit is not None:
        transcript_paths = transcript_paths[:limit]

    if not transcript_paths:
        msg = f"No transcript files found in {transcripts_dir}"
        raise ValueError(msg)

    return [load_transcript(path) for path in transcript_paths]
