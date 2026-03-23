"""FAISS-backed vector index helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from neural.chunking import ChunkingConfig, TranscriptChunk
from neural.embeddings import EMBEDDING_BACKEND_OPENROUTER

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
CONFIG_FILENAME = "config.json"
SOURCE_MANIFEST_FILENAME = "source_manifest.json"

ARTIFACT_VERSION_V1 = 1
ARTIFACT_VERSION_V2 = 2


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


def _is_idmap_index(index: Any) -> bool:
    return "IDMap" in type(index).__name__


def build_faiss_index(embeddings: np.ndarray):
    """Build an exact-search FAISS index with stable row ids (IndexIDMap2 + IndexFlatIP).

    Ids are ``0 .. n-1``, matching contiguous ``chunk_id`` assignments for new indexes.
    """
    faiss = _require_faiss()
    matrix = _coerce_embeddings(embeddings)
    n, dim = matrix.shape
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    ids = np.arange(n, dtype=np.int64)
    index.add_with_ids(matrix, ids)
    return index


def build_faiss_index_flat_legacy(embeddings: np.ndarray):
    """Build a legacy IndexFlatIP without explicit vector ids (artifact v1 only)."""
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


def _assign_contiguous_chunk_ids(chunks: list[TranscriptChunk]) -> list[TranscriptChunk]:
    return [replace(c, chunk_id=i) for i, c in enumerate(chunks)]


def save_index_artifacts(
    *,
    output_dir: Path,
    index,
    chunks: list[TranscriptChunk],
    model_name: str,
    chunking_config: ChunkingConfig,
    transcripts_dir: Path,
    source_manifest: dict[str, str] | None = None,
    embedding_backend: str = EMBEDDING_BACKEND_OPENROUTER,
) -> None:
    """Persist index, chunk metadata, and build configuration."""
    faiss = _require_faiss()
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_with_ids: list[TranscriptChunk] = []
    for i, chunk in enumerate(chunks):
        cid = chunk.chunk_id if chunk.chunk_id is not None else i
        chunks_with_ids.append(replace(chunk, chunk_id=cid))

    faiss.write_index(index, str(output_dir / INDEX_FILENAME))
    (output_dir / CHUNKS_FILENAME).write_text(
        json.dumps([chunk.to_dict() for chunk in chunks_with_ids], indent=2),
        encoding="utf-8",
    )

    config: dict[str, Any] = {
        "model_name": model_name,
        "embedding_backend": embedding_backend,
        "transcripts_dir": str(transcripts_dir),
        "chunking": chunking_config.to_dict(),
        "artifact_version": ARTIFACT_VERSION_V2,
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    if source_manifest is not None:
        (output_dir / SOURCE_MANIFEST_FILENAME).write_text(
            json.dumps(dict(sorted(source_manifest.items())), indent=2),
            encoding="utf-8",
        )


def load_source_manifest(index_dir: Path) -> dict[str, str]:
    path = index_dir / SOURCE_MANIFEST_FILENAME
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "source_manifest.json must contain a JSON object"
        raise ValueError(msg)
    return {str(k): str(v) for k, v in data.items()}


def remove_ids_from_index(index, ids_to_remove: np.ndarray) -> int:
    """Remove FAISS vectors by id. Index must be IndexIDMap2."""
    if ids_to_remove.size == 0:
        return 0
    faiss = _require_faiss()
    if not _is_idmap_index(index):
        msg = "remove_ids requires an IndexIDMap2-backed index"
        raise ValueError(msg)
    ids64 = np.ascontiguousarray(ids_to_remove.astype(np.int64))
    sel = faiss.IDSelectorBatch(int(ids64.size), faiss.swig_ptr(ids64))
    return int(index.remove_ids(sel))


def add_vectors_with_ids(index, embeddings: np.ndarray, ids: np.ndarray) -> None:
    """Append normalized embeddings with explicit int64 ids (IndexIDMap2)."""
    if not _is_idmap_index(index):
        msg = "add_vectors_with_ids requires an IndexIDMap2-backed index"
        raise ValueError(msg)
    matrix = _coerce_embeddings(embeddings)
    ids64 = np.ascontiguousarray(ids.astype(np.int64))
    if matrix.shape[0] != ids64.shape[0]:
        msg = "embeddings and ids must have the same length"
        raise ValueError(msg)
    index.add_with_ids(matrix, ids64)


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
    chunks = [_chunk_from_dict(d) for d in chunk_dicts]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    version = int(config.get("artifact_version", ARTIFACT_VERSION_V1))

    if _is_idmap_index(index):
        chunks = _ensure_chunk_ids(chunks, version)
    else:
        chunks = _assign_contiguous_chunk_ids(
            [replace(c, chunk_id=None) for c in chunks],
        )

    return index, chunks, config


def _chunk_from_dict(d: dict[str, Any]) -> TranscriptChunk:
    payload = dict(d)
    if "chunk_id" not in payload:
        payload["chunk_id"] = None
    return TranscriptChunk(**payload)


def _ensure_chunk_ids(chunks: list[TranscriptChunk], version: int) -> list[TranscriptChunk]:
    if all(c.chunk_id is not None for c in chunks):
        return chunks
    if version >= ARTIFACT_VERSION_V2:
        msg = "artifact_version >= 2 requires chunk_id on every chunk"
        raise ValueError(msg)
    return _assign_contiguous_chunk_ids(chunks)


def build_search_results(
    scores: np.ndarray,
    indices: np.ndarray,
    chunks: list[TranscriptChunk],
) -> list[SearchResult]:
    """Convert raw ranked ids into structured retrieval results."""
    id_to_chunk: dict[int, TranscriptChunk] = {}
    for chunk in chunks:
        if chunk.chunk_id is None:
            msg = "Every chunk must have chunk_id before build_search_results"
            raise ValueError(msg)
        id_to_chunk[chunk.chunk_id] = chunk

    results: list[SearchResult] = []
    flat_scores = np.asarray(scores).flatten()
    flat_indices = np.asarray(indices).flatten()

    pairs = zip(flat_scores, flat_indices, strict=True)
    for offset, (score, index_id) in enumerate(pairs, start=1):
        if int(index_id) < 0:
            continue
        chunk = id_to_chunk.get(int(index_id))
        if chunk is None:
            continue
        results.append(
            SearchResult(
                rank=offset,
                score=float(score),
                chunk=chunk,
            )
        )
    return results
