"""Utilities for transcript retrieval experiments."""

from __future__ import annotations

from typing import Any

from neural.chunking import ChunkingConfig, TranscriptChunk, chunk_corpus, chunk_transcript
from neural.corpus import TranscriptDocument, TranscriptLine, load_corpus, load_transcript

# Retrieval pulls in embeddings / numpy; load lazily so `import neural.chunk_enrichment`
# (and similar) stays lightweight and avoids duplicate native extension loads under
# pytest-cov on Windows.
_RETRIEVAL_EXPORTS: frozenset[str] = frozenset(
    {
        "RetrievalBundle",
        "citations_from_results",
        "load_retrieval_bundle",
        "retrieve",
        "retrieve_from_disk",
        "search_result_to_citation",
    }
)

__all__ = [
    "ChunkingConfig",
    "RetrievalBundle",
    "TranscriptChunk",
    "TranscriptDocument",
    "TranscriptLine",
    "chunk_corpus",
    "chunk_transcript",
    "citations_from_results",
    "load_corpus",
    "load_retrieval_bundle",
    "load_transcript",
    "retrieve",
    "retrieve_from_disk",
    "search_result_to_citation",
]


def __getattr__(name: str) -> Any:
    if name in _RETRIEVAL_EXPORTS:
        from neural import retrieval

        return getattr(retrieval, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
