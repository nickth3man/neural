"""Tests for vector indexing helpers."""

from pathlib import Path

import numpy as np
import pytest
import torch

from neural.chunking import ChunkingConfig, TranscriptChunk
from neural.vector_index import (
    build_faiss_index,
    build_search_results,
    cosine_search,
    load_index_bundle,
    save_index_artifacts,
    search_index,
)

faiss = pytest.importorskip("faiss")


def _normalized(array: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(array, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def test_faiss_search_matches_util_semantic_search_order() -> None:
    st_util = pytest.importorskip("sentence_transformers.util")
    embeddings = _normalized(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    query_embedding = _normalized([[0.9, 0.8]])
    index = build_faiss_index(embeddings)

    faiss_scores, faiss_ids = search_index(index, query_embedding, top_k=3)

    corpus_t = torch.from_numpy(embeddings)
    query_t = torch.from_numpy(query_embedding.reshape(1, -1))
    hits = st_util.semantic_search(query_t, corpus_t, top_k=3)[0]
    util_ids = [h["corpus_id"] for h in hits]

    assert faiss_ids.flatten().tolist() == util_ids
    assert faiss_scores.flatten().tolist() == pytest.approx([h["score"] for h in hits])


def test_faiss_search_matches_in_memory_cosine_search() -> None:
    embeddings = _normalized(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    query_embedding = _normalized([[0.9, 0.8]])
    index = build_faiss_index(embeddings)

    faiss_scores, faiss_ids = search_index(index, query_embedding, top_k=3)
    cosine_scores, cosine_ids = cosine_search(query_embedding, embeddings, top_k=3)

    assert faiss_ids.tolist()[0] == cosine_ids.tolist()
    assert faiss_scores.tolist()[0] == pytest.approx(cosine_scores.tolist())


def test_save_and_load_index_bundle_round_trip(tmp_path: Path) -> None:
    embeddings = _normalized([[1.0, 0.0], [0.0, 1.0]])
    index = build_faiss_index(embeddings)
    chunks = [
        TranscriptChunk(
            episode_title="Episode One",
            source_file="episode_one.txt",
            start_timestamp="00:00:01",
            end_timestamp="00:00:05",
            start_seconds=1,
            end_seconds=5,
            chunk_text="First chunk text",
            line_count=2,
        ),
        TranscriptChunk(
            episode_title="Episode Two",
            source_file="episode_two.txt",
            start_timestamp="00:00:06",
            end_timestamp="00:00:10",
            start_seconds=6,
            end_seconds=10,
            chunk_text="Second chunk text",
            line_count=2,
        ),
    ]

    save_index_artifacts(
        output_dir=tmp_path,
        index=index,
        chunks=chunks,
        model_name="all-MiniLM-L6-v2",
        chunking_config=ChunkingConfig(),
        transcripts_dir=Path("gil/transcripts"),
    )

    loaded_index, loaded_chunks, config = load_index_bundle(tmp_path)

    assert loaded_index.ntotal == 2
    assert loaded_chunks == chunks
    assert config["model_name"] == "all-MiniLM-L6-v2"
    assert config["chunking"]["lines_per_chunk"] == 5


def test_build_search_results_preserves_rank_and_metadata() -> None:
    chunks = [
        TranscriptChunk(
            episode_title="Episode One",
            source_file="episode_one.txt",
            start_timestamp="00:00:01",
            end_timestamp="00:00:05",
            start_seconds=1,
            end_seconds=5,
            chunk_text="First chunk text",
            line_count=2,
        ),
        TranscriptChunk(
            episode_title="Episode Two",
            source_file="episode_two.txt",
            start_timestamp="00:00:06",
            end_timestamp="00:00:10",
            start_seconds=6,
            end_seconds=10,
            chunk_text="Second chunk text",
            line_count=2,
        ),
    ]

    results = build_search_results(np.array([[0.9, 0.4]]), np.array([[1, 0]]), chunks)

    assert [result.rank for result in results] == [1, 2]
    assert results[0].chunk.episode_title == "Episode Two"
    assert results[1].chunk.source_file == "episode_one.txt"
