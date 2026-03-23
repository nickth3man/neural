"""Tests for BM25 + RRF hybrid retrieval helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neural.chunking import TranscriptChunk
from neural.embeddings import EMBEDDING_BACKEND_OPENROUTER
from neural.hybrid_retrieval import bm25_ranked_chunk_ids, reciprocal_rank_fusion, tokenize_for_bm25
from neural.retrieval import RetrievalBundle, retrieve

faiss = pytest.importorskip("faiss")


def test_tokenize_for_bm25_alphanumeric() -> None:
    assert tokenize_for_bm25("Hello, XYZZY-123!") == ["hello", "xyzzy", "123"]


def test_reciprocal_rank_fusion_orders_by_rrf_score() -> None:
    fused = reciprocal_rank_fusion([[10, 20], [20, 30]], rrf_k=60)
    ids = [x[0] for x in fused]
    assert 20 in ids
    assert ids[0] == 20


def test_retrieve_hybrid_prefers_lexical_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dense favors chunk A; BM25 ranks chunk B first for a term only B contains."""
    from neural import retrieval as retrieval_mod

    chunks = [
        TranscriptChunk(
            episode_title="A",
            source_file="a.txt",
            start_timestamp="00:00:01",
            end_timestamp="00:00:02",
            start_seconds=1,
            end_seconds=2,
            chunk_text="common words about basketball games and teams playing well",
            line_count=2,
            chunk_id=0,
        ),
        TranscriptChunk(
            episode_title="B",
            source_file="b.txt",
            start_timestamp="00:00:10",
            end_timestamp="00:00:12",
            start_seconds=10,
            end_seconds=12,
            chunk_text="unrelated intro about football tactics and defense schemes here",
            line_count=2,
            chunk_id=1,
        ),
        TranscriptChunk(
            episode_title="C",
            source_file="c.txt",
            start_timestamp="00:00:20",
            end_timestamp="00:00:22",
            start_seconds=20,
            end_seconds=22,
            chunk_text="filler episode discussing weather patterns and seasonal changes today",
            line_count=2,
            chunk_id=2,
        ),
    ]

    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    from neural.vector_index import build_faiss_index

    index = build_faiss_index(embeddings)
    bundle = RetrievalBundle(
        index=index,
        chunks=chunks,
        config={"model_name": "fixture", "embedding_backend": EMBEDDING_BACKEND_OPENROUTER},
    )

    def fake_encode(texts: list[str], model_name: str = "", **_: object) -> np.ndarray:
        # Fixed query direction so dense similarity always favors chunk A (id 0).
        row = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        return np.repeat(row, len(texts), axis=0)

    monkeypatch.setattr(retrieval_mod, "encode_texts", fake_encode)

    dense_only = retrieve(bundle, "football tactics defense", top_k=1, hybrid=False)
    assert dense_only[0].chunk.source_file == "a.txt"

    hybrid = retrieve(bundle, "football tactics defense", top_k=1, hybrid=True, hybrid_lexical_k=1)
    assert hybrid[0].chunk.source_file == "b.txt"


def test_bm25_ranked_chunk_ids_respects_top_n() -> None:
    chunks = [
        TranscriptChunk(
            episode_title="E",
            source_file=f"{i}.txt",
            start_timestamp="00:00:01",
            end_timestamp="00:00:02",
            start_seconds=1,
            end_seconds=2,
            chunk_text="alpha beta gamma" if i else "delta epsilon zeta uniqueword",
            line_count=2,
            chunk_id=i,
        )
        for i in range(4)
    ]
    ids = bm25_ranked_chunk_ids(chunks, "uniqueword", top_n=2)
    assert 0 in ids or 1 in ids or 2 in ids or 3 in ids
    assert len(ids) <= 2
