"""Tests for cross-encoder reranking (mocked model)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

import neural.reranking as reranking_mod
from neural.chunking import TranscriptChunk
from neural.reranking import RerankerConfig, rerank_results
from neural.vector_index import SearchResult


def _chunk(text: str) -> TranscriptChunk:
    return TranscriptChunk(
        episode_title="E",
        source_file="e.txt",
        start_timestamp="00:00:01",
        end_timestamp="00:00:02",
        start_seconds=1,
        end_seconds=2,
        chunk_text=text,
        line_count=1,
    )


def test_rerank_results_empty() -> None:
    assert rerank_results("q", [], RerankerConfig()) == []


@patch("neural.reranking._load_cross_encoder")
def test_rerank_results_reorders_by_score(mock_load: MagicMock) -> None:
    reranking_mod._load_cross_encoder.cache_clear()

    model = MagicMock()
    model.predict.return_value = [0.2, 0.8]
    mock_load.return_value = model

    a = SearchResult(rank=1, score=0.9, chunk=_chunk("first"))
    b = SearchResult(rank=2, score=0.1, chunk=_chunk("second"))
    out = rerank_results("query", [a, b], RerankerConfig(top_n=20))

    assert len(out) == 2
    assert out[0].chunk.chunk_text == "second"
    assert out[0].rank == 1
    assert out[0].score == 0.8
    assert out[1].chunk.chunk_text == "first"
    assert out[1].rank == 2
    model.predict.assert_called_once()
    reranking_mod._load_cross_encoder.cache_clear()


@patch("neural.reranking._load_cross_encoder")
def test_rerank_results_top_n_appends_tail(mock_load: MagicMock) -> None:
    reranking_mod._load_cross_encoder.cache_clear()

    model = MagicMock()
    model.predict.return_value = [1.0]
    mock_load.return_value = model

    chunks = [_chunk(f"c{i}") for i in range(4)]
    results = [SearchResult(rank=i + 1, score=float(i), chunk=chunks[i]) for i in range(4)]
    out = rerank_results("q", results, RerankerConfig(top_n=1))

    assert len(out) == 4
    assert out[0].chunk.chunk_text == "c0"
    assert out[1].rank == 2
    assert out[1].score == 1.0
    assert out[1].chunk.chunk_text == "c1"
    reranking_mod._load_cross_encoder.cache_clear()


def test_load_cross_encoder_import_error() -> None:
    reranking_mod._load_cross_encoder.cache_clear()
    with patch.object(importlib, "import_module", side_effect=ImportError("no st")):
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            reranking_mod._load_cross_encoder("any-model")
    reranking_mod._load_cross_encoder.cache_clear()
