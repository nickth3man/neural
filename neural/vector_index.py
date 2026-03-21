"""FAISS-backed vector index helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural.chunking import ChunkingConfig, TranscriptChunk

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class SearchResult:
    """One ranked retrieval hit."""

    rank: int
    score: float
    chunk: TranscriptChunk


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:
        msg = (
            "faiss is required for vector indexing. Install project dependencies "
            "before building or querying the index."
        )
        raise RuntimeError(msg) from exc
    return faiss


def _coerce_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype="float32")
    if array.ndim != 2:
        msg = "Embeddings must be a 2D matrix"
        raise ValueError(msg)
    if array.shape[0] == 0:
        msg = "Embeddings matrix cannot be empty"
        raise ValueError(msg)
    return array


def build_faiss_index(embeddings: np.ndarray):
    """Build an exact-search FAISS index from normalized embeddings."""
    faiss = _require_faiss()
    matrix = _coerce_embeddings(embeddings)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def cosine_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search normalized embeddings using pure in-memory cosine similarity."""
    if top_k < 1:
        msg = "top_k must be at least 1"
        raise ValueError(msg)

    matrix = _coerce_embeddings(embeddings)
    query = np.asarray(query_embedding, dtype="float32")
    if query.ndim == 2:
        if query.shape[0] != 1:
            msg = "Query embedding must contain exactly one row"
            raise ValueError(msg)
        query = query[0]
    if query.ndim != 1:
        msg = "Query embedding must be 1D or a single-row 2D array"
        raise ValueError(msg)

    scores = matrix @ query
    limited_top_k = min(top_k, len(scores))
    indices = np.argsort(-scores)[:limited_top_k]
    return scores[indices], indices


def search_index(index, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search a FAISS index using one normalized query embedding."""
    if top_k < 1:
        msg = "top_k must be at least 1"
        raise ValueError(msg)

    query = np.asarray(query_embedding, dtype="float32")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[0] != 1:
        msg = "Query embedding must be 1D or a single-row 2D array"
        raise ValueError(msg)

    limited_top_k = min(top_k, index.ntotal)
    return index.search(query, limited_top_k)


def save_index_artifacts(
    *,
    output_dir: Path,
    index,
    chunks: list[TranscriptChunk],
    model_name: str,
    chunking_config: ChunkingConfig,
    transcripts_dir: Path,
) -> None:
    """Persist index, chunk metadata, and build configuration."""
    faiss = _require_faiss()
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / INDEX_FILENAME))
    (output_dir / CHUNKS_FILENAME).write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2),
        encoding="utf-8",
    )

    config: dict[str, Any] = {
        "model_name": model_name,
        "transcripts_dir": str(transcripts_dir),
        "chunking": chunking_config.to_dict(),
        "artifact_version": 1,
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def load_index_bundle(index_dir: Path):
    """Load a previously saved FAISS index bundle."""
    faiss = _require_faiss()
    index_path = index_dir / INDEX_FILENAME
    chunks_path = index_dir / CHUNKS_FILENAME
    config_path = index_dir / CONFIG_FILENAME

    for artifact_path in (index_path, chunks_path, config_path):
        if not artifact_path.exists():
            msg = f"Missing index artifact: {artifact_path}"
            raise FileNotFoundError(msg)

    index = faiss.read_index(str(index_path))
    chunk_dicts = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [TranscriptChunk(**chunk_dict) for chunk_dict in chunk_dicts]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return index, chunks, config


def build_search_results(
    scores: np.ndarray,
    indices: np.ndarray,
    chunks: list[TranscriptChunk],
) -> list[SearchResult]:
    """Convert raw ranked ids into structured retrieval results."""
    results: list[SearchResult] = []
    flat_scores = np.asarray(scores).flatten()
    flat_indices = np.asarray(indices).flatten()

    pairs = zip(flat_scores, flat_indices, strict=True)
    for offset, (score, index_id) in enumerate(pairs, start=1):
        if index_id < 0:
            continue
        results.append(
            SearchResult(
                rank=offset,
                score=float(score),
                chunk=chunks[int(index_id)],
            )
        )
    return results
