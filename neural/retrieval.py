"""Shared transcript retrieval over a loaded FAISS index bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural.chunking import TranscriptChunk
from neural.embeddings import encode_texts
from neural.vector_index import SearchResult, build_search_results, load_index_bundle, search_index


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    """In-memory index, chunk metadata, and build config from disk."""

    index: Any
    chunks: list[TranscriptChunk]
    config: dict[str, Any]


def load_retrieval_bundle(index_dir: Path) -> RetrievalBundle:
    """Load ``index.faiss``, ``chunks.json``, and ``config.json`` from ``index_dir``."""
    index, chunks, config = load_index_bundle(index_dir)
    return RetrievalBundle(index=index, chunks=chunks, config=config)


def retrieve(
    bundle: RetrievalBundle,
    query: str,
    *,
    top_k: int = 5,
    model_override: str | None = None,
) -> list[SearchResult]:
    """
    Embed ``query``, search the bundle index, and return ranked ``SearchResult`` rows.

    Uses ``model_override`` when set; otherwise ``bundle.config["model_name"]``.
    """
    if top_k < 1:
        msg = "top_k must be at least 1"
        raise ValueError(msg)

    model_name = model_override or bundle.config["model_name"]
    query_embedding = encode_texts([query], model_name=model_name)
    scores, indices = search_index(bundle.index, query_embedding, top_k)
    return build_search_results(scores, indices, bundle.chunks)


def retrieve_from_disk(
    query: str,
    index_dir: Path,
    *,
    top_k: int = 5,
    model_override: str | None = None,
) -> list[SearchResult]:
    """Load bundle from disk and retrieve in one call (CLI convenience)."""
    bundle = load_retrieval_bundle(index_dir)
    return retrieve(bundle, query, top_k=top_k, model_override=model_override)


def search_result_to_citation(result: SearchResult) -> dict[str, str | int | float]:
    """JSON-serializable citation dict for APIs and UI."""
    chunk = result.chunk
    return {
        "rank": result.rank,
        "score": round(float(result.score), 6),
        "episode_title": chunk.episode_title,
        "source_file": chunk.source_file,
        "start_timestamp": chunk.start_timestamp,
        "end_timestamp": chunk.end_timestamp,
        "chunk_text": chunk.chunk_text,
    }


def citations_from_results(results: list[SearchResult]) -> list[dict[str, str | int | float]]:
    """Map a result list to citation dicts."""
    return [search_result_to_citation(r) for r in results]
