"""Shared transcript retrieval over a loaded FAISS index bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural.chunking import TranscriptChunk
from neural.embeddings import (
    EMBEDDING_BACKEND_OPENROUTER,
    LEGACY_EMBEDDING_BACKEND_SENTENCE_TRANSFORMERS,
    encode_texts,
)
from neural.hybrid_retrieval import bm25_ranked_chunk_ids, reciprocal_rank_fusion
from neural.metadata_index import (
    MetadataIndex,
    RetrievalFilters,
    filter_search_results,
    metadata_to_citation,
)
from neural.reranking import RerankerConfig, rerank_results
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
    metadata_index: MetadataIndex | None = None,
    filters: RetrievalFilters | None = None,
    reranker: RerankerConfig | None = None,
    hybrid: bool = False,
    hybrid_lexical_k: int = 20,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """
    Embed ``query``, search the bundle index, and return ranked ``SearchResult`` rows.

    Uses ``model_override`` when set; otherwise ``bundle.config["model_name"]``.
    """
    if top_k < 1:
        msg = "top_k must be at least 1"
        raise ValueError(msg)

    model_name = model_override or bundle.config["model_name"]
    backend = bundle.config.get(
        "embedding_backend",
        LEGACY_EMBEDDING_BACKEND_SENTENCE_TRANSFORMERS,
    )
    if backend != EMBEDDING_BACKEND_OPENROUTER:
        msg = (
            "This index was built with local sentence-transformers embeddings or predates "
            "embedding_backend in config. Rebuild the index with OpenRouter embeddings "
            "(set OPENROUTER_EMBEDDING_MODEL and OPENROUTER_API_KEY, then run a full index build)."
        )
        raise ValueError(msg)
    query_embedding = encode_texts([query], model_name=model_name)
    candidate_count = top_k
    if reranker is not None:
        candidate_count = max(candidate_count, reranker.top_n)
    if filters is not None and filters.active() and metadata_index is not None:
        candidate_count = max(candidate_count, top_k * 10)
    if hybrid:
        # Need multiple dense hits so RRF can combine with BM25; top-1 dense + BM25 ties otherwise.
        candidate_count = max(candidate_count, hybrid_lexical_k, top_k + hybrid_lexical_k)

    if hybrid:
        scores, indices = search_index(bundle.index, query_embedding, candidate_count)
        dense_results = build_search_results(scores, indices, bundle.chunks)
        dense_ids = [int(r.chunk.chunk_id) for r in dense_results if r.chunk.chunk_id is not None]
        bm25_ids = bm25_ranked_chunk_ids(bundle.chunks, query, hybrid_lexical_k)
        rankings = [dense_ids]
        if bm25_ids:
            rankings.append(bm25_ids)
        fused = reciprocal_rank_fusion(rankings, rrf_k=rrf_k)
        id_to_chunk = {int(c.chunk_id): c for c in bundle.chunks if c.chunk_id is not None}
        results = []
        for offset, (cid, rrf_score) in enumerate(fused[:candidate_count], start=1):
            chunk = id_to_chunk.get(cid)
            if chunk is None:
                continue
            results.append(
                SearchResult(rank=offset, score=float(rrf_score), chunk=chunk),
            )
    else:
        scores, indices = search_index(bundle.index, query_embedding, candidate_count)
        results = build_search_results(scores, indices, bundle.chunks)

    if filters is not None and filters.active() and metadata_index is not None:
        results = filter_search_results(results, metadata_index, filters)
    if reranker is not None:
        results = rerank_results(query, results, reranker)

    ranked: list[SearchResult] = []
    for rank, result in enumerate(results[:top_k], start=1):
        ranked.append(SearchResult(rank=rank, score=result.score, chunk=result.chunk))
    return ranked


def retrieve_from_disk(
    query: str,
    index_dir: Path,
    *,
    top_k: int = 5,
    model_override: str | None = None,
    metadata_index: MetadataIndex | None = None,
    filters: RetrievalFilters | None = None,
    reranker: RerankerConfig | None = None,
    hybrid: bool = False,
    hybrid_lexical_k: int = 20,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Load bundle from disk and retrieve in one call (CLI convenience)."""
    bundle = load_retrieval_bundle(index_dir)
    return retrieve(
        bundle,
        query,
        top_k=top_k,
        model_override=model_override,
        metadata_index=metadata_index,
        filters=filters,
        reranker=reranker,
        hybrid=hybrid,
        hybrid_lexical_k=hybrid_lexical_k,
        rrf_k=rrf_k,
    )


def search_result_to_citation(
    result: SearchResult,
    metadata_index: MetadataIndex | None = None,
) -> dict[str, object]:
    """JSON-serializable citation dict for APIs and UI."""
    chunk = result.chunk
    citation: dict[str, object] = {
        "rank": result.rank,
        "score": round(float(result.score), 6),
        "episode_title": chunk.episode_title,
        "source_file": chunk.source_file,
        "start_timestamp": chunk.start_timestamp,
        "end_timestamp": chunk.end_timestamp,
        "chunk_text": chunk.chunk_text,
    }
    metadata = (
        metadata_to_citation(metadata_index.metadata_for_result(result)) if metadata_index else None
    )
    if metadata is not None:
        citation["metadata"] = metadata
    return citation


def citations_from_results(
    results: list[SearchResult],
    metadata_index: MetadataIndex | None = None,
) -> list[dict[str, object]]:
    """Map a result list to citation dicts."""
    return [search_result_to_citation(r, metadata_index) for r in results]
