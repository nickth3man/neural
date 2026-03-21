"""Transcript chunking helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from neural.corpus import TranscriptDocument, format_timestamp, normalize_text


@dataclass(frozen=True)
class ChunkingConfig:
    """Deterministic line-window chunking configuration."""

    lines_per_chunk: int = 5
    line_overlap: int = 1
    min_chunk_characters: int = 40

    def __post_init__(self) -> None:
        if self.lines_per_chunk < 1:
            msg = "lines_per_chunk must be at least 1"
            raise ValueError(msg)
        if self.line_overlap < 0:
            msg = "line_overlap cannot be negative"
            raise ValueError(msg)
        if self.line_overlap >= self.lines_per_chunk:
            msg = "line_overlap must be smaller than lines_per_chunk"
            raise ValueError(msg)
        if self.min_chunk_characters < 1:
            msg = "min_chunk_characters must be at least 1"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable configuration mapping."""
        return asdict(self)


@dataclass(frozen=True)
class TranscriptChunk:
    """One retrieval unit derived from transcript lines."""

    episode_title: str
    source_file: str
    start_timestamp: str
    end_timestamp: str
    start_seconds: int
    end_seconds: int
    chunk_text: str
    line_count: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable chunk mapping."""
        return asdict(self)


def chunk_transcript(
    document: TranscriptDocument,
    config: ChunkingConfig = ChunkingConfig(),
) -> list[TranscriptChunk]:
    """Split a transcript document into overlapping retrieval chunks."""
    if not document.lines:
        return []

    step = config.lines_per_chunk - config.line_overlap
    lines = list(document.lines)
    total_lines = len(lines)
    chunks: list[TranscriptChunk] = []
    start = 0

    while start < total_lines:
        end = min(start + config.lines_per_chunk, total_lines)
        window = lines[start:end]
        if len(window) < 2 and end == total_lines and chunks:
            break

        chunk_text = normalize_text(" ".join(line.text for line in window if line.text))
        if len(window) >= 2 and len(chunk_text) >= config.min_chunk_characters:
            first = window[0]
            last = window[-1]
            chunks.append(
                TranscriptChunk(
                    episode_title=document.episode_title,
                    source_file=document.source_file,
                    start_timestamp=format_timestamp(first.start_seconds),
                    end_timestamp=format_timestamp(last.start_seconds),
                    start_seconds=first.start_seconds,
                    end_seconds=last.start_seconds,
                    chunk_text=chunk_text,
                    line_count=len(window),
                )
            )

        if end == total_lines:
            break
        start += step

    return chunks


def chunk_corpus(
    documents: list[TranscriptDocument],
    config: ChunkingConfig = ChunkingConfig(),
) -> list[TranscriptChunk]:
    """Chunk every transcript document in the corpus."""
    chunks: list[TranscriptChunk] = []
    for document in documents:
        chunks.extend(chunk_transcript(document, config))
    return chunks
