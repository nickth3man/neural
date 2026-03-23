"""Tests for shared retrieval service (neural.retrieval)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural.chunking import ChunkingConfig, TranscriptChunk
from neural.embeddings import LEGACY_EMBEDDING_BACKEND_SENTENCE_TRANSFORMERS
from neural.metadata_index import RetrievalFilters, load_metadata_index
from neural.reranking import RerankerConfig
from neural.retrieval import (
    RetrievalBundle,
    citations_from_results,
    load_retrieval_bundle,
    retrieve,
    retrieve_from_disk,
    search_result_to_citation,
)
from neural.vector_index import SearchResult, build_faiss_index, save_index_artifacts

faiss = pytest.importorskip("faiss")


def _normalized(rows: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def _write_min_bundle(tmp_path: Path) -> None:
    embeddings = _normalized([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    index = build_faiss_index(embeddings)
    chunks = [
        TranscriptChunk(
            episode_title="Episode_A",
            source_file="a.txt",
            start_timestamp="00:00:01",
            end_timestamp="00:00:02",
            start_seconds=1,
            end_seconds=2,
            chunk_text="alpha chunk",
            line_count=2,
        ),
        TranscriptChunk(
            episode_title="Episode_B",
            source_file="b.txt",
            start_timestamp="00:00:10",
            end_timestamp="00:00:12",
            start_seconds=10,
            end_seconds=12,
            chunk_text="beta chunk",
            line_count=2,
        ),
    ]
    save_index_artifacts(
        output_dir=tmp_path,
        index=index,
        chunks=chunks,
        model_name="fixture-model",
        chunking_config=ChunkingConfig(),
        transcripts_dir=Path("gil/transcripts"),
    )


def _write_metadata(tmp_path: Path) -> Path:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "enriched_chunks.json").write_text(
        json.dumps(
            [
                {
                    "source_file": "a.txt",
                    "start_timestamp": "00:00:01",
                    "end_timestamp": "00:00:02",
                    "episode_type": "reaction",
                    "guest_names": ["Guest A"],
                    "host_names": ["Host A"],
                    "episode_topic": "alpha",
                    "chunk_meta": {
                        "mentioned_people": ["Alpha Person"],
                        "mentioned_teams": ["Lakers"],
                        "mentioned_leagues": ["NBA"],
                    },
                    "llm_meta": {"speaker": "Speaker A", "topic": "alpha", "subtopic": "trade"},
                },
                {
                    "source_file": "b.txt",
                    "start_timestamp": "00:00:10",
                    "end_timestamp": "00:00:12",
                    "episode_type": "guest_interview",
                    "guest_names": ["Guest B"],
                    "host_names": ["Host B"],
                    "episode_topic": "beta",
                    "chunk_meta": {
                        "mentioned_people": ["Beta Person"],
                        "mentioned_teams": ["Celtics"],
                        "mentioned_leagues": ["NBA"],
                    },
                    "llm_meta": {"speaker": "Speaker B", "topic": "beta", "subtopic": "playoff"},
                },
            ]
        ),
        encoding="utf-8",
    )
    (metadata_dir / "document_metadata.json").write_text(
        json.dumps(
            [
                {
                    "source_file": "a.txt",
                    "episode_type": "reaction",
                    "guest_names": ["Guest A"],
                    "host_names": ["Host A"],
                    "topic": "alpha",
                    "holiday_theme": "",
                },
                {
                    "source_file": "b.txt",
                    "episode_type": "guest_interview",
                    "guest_names": ["Guest B"],
                    "host_names": ["Host B"],
                    "topic": "beta",
                    "holiday_theme": "",
                },
            ]
        ),
        encoding="utf-8",
    )
    return metadata_dir


def test_load_retrieval_bundle_round_trip(tmp_path: Path) -> None:
    _write_min_bundle(tmp_path)
    bundle = load_retrieval_bundle(tmp_path)
    assert isinstance(bundle, RetrievalBundle)
    assert bundle.config["model_name"] == "fixture-model"
    assert len(bundle.chunks) == 2


def test_retrieve_invalid_top_k(tmp_path: Path) -> None:
    _write_min_bundle(tmp_path)
    bundle = load_retrieval_bundle(tmp_path)
    with pytest.raises(ValueError, match="top_k"):
        retrieve(bundle, "q", top_k=0)


def test_retrieve_rejects_legacy_sentence_transformers_index(tmp_path: Path) -> None:
    embeddings = _normalized([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    index = build_faiss_index(embeddings)
    chunks = [
        TranscriptChunk(
            episode_title="Episode_A",
            source_file="a.txt",
            start_timestamp="00:00:01",
            end_timestamp="00:00:02",
            start_seconds=1,
            end_seconds=2,
            chunk_text="alpha chunk",
            line_count=2,
        ),
        TranscriptChunk(
            episode_title="Episode_B",
            source_file="b.txt",
            start_timestamp="00:00:10",
            end_timestamp="00:00:12",
            start_seconds=10,
            end_seconds=12,
            chunk_text="beta chunk",
            line_count=2,
        ),
    ]
    save_index_artifacts(
        output_dir=tmp_path,
        index=index,
        chunks=chunks,
        model_name="fixture-model",
        chunking_config=ChunkingConfig(),
        transcripts_dir=Path("gil/transcripts"),
        embedding_backend=LEGACY_EMBEDDING_BACKEND_SENTENCE_TRANSFORMERS,
    )
    bundle = load_retrieval_bundle(tmp_path)
    with pytest.raises(ValueError, match="Rebuild the index"):
        retrieve(bundle, "q", top_k=1)


def test_retrieve_rejects_index_missing_embedding_backend_key(tmp_path: Path) -> None:
    """Predates embedding_backend field; treated as legacy local index."""
    embeddings = _normalized([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    index = build_faiss_index(embeddings)
    chunks = [
        TranscriptChunk(
            episode_title="Episode_A",
            source_file="a.txt",
            start_timestamp="00:00:01",
            end_timestamp="00:00:02",
            start_seconds=1,
            end_seconds=2,
            chunk_text="alpha chunk",
            line_count=2,
        ),
        TranscriptChunk(
            episode_title="Episode_B",
            source_file="b.txt",
            start_timestamp="00:00:10",
            end_timestamp="00:00:12",
            start_seconds=10,
            end_seconds=12,
            chunk_text="beta chunk",
            line_count=2,
        ),
    ]
    save_index_artifacts(
        output_dir=tmp_path,
        index=index,
        chunks=chunks,
        model_name="fixture-model",
        chunking_config=ChunkingConfig(),
        transcripts_dir=Path("gil/transcripts"),
    )
    config_path = tmp_path / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    del cfg["embedding_backend"]
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    bundle = load_retrieval_bundle(tmp_path)
    with pytest.raises(ValueError, match="Rebuild the index"):
        retrieve(bundle, "q", top_k=1)


def test_retrieve_uses_config_model_and_ranking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_min_bundle(tmp_path)
    bundle = load_retrieval_bundle(tmp_path)

    def fake_encode(
        texts: list[str],
        *,
        model_name: str = "fixture-model",
        normalize_embeddings: bool = True,
        **_: object,
    ) -> np.ndarray:
        assert model_name == "fixture-model"
        assert texts == ["find alpha"]
        return np.asarray([[1.0, 0.0, 0.0]], dtype="float32")

    monkeypatch.setattr("neural.retrieval.encode_texts", fake_encode)
    results = retrieve(bundle, "find alpha", top_k=2)
    assert results[0].chunk.chunk_text == "alpha chunk"
    assert results[0].rank == 1


def test_retrieve_model_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_min_bundle(tmp_path)
    bundle = load_retrieval_bundle(tmp_path)

    def fake_encode(
        texts: list[str],
        *,
        model_name: str,
        normalize_embeddings: bool = True,
        **_: object,
    ) -> np.ndarray:
        assert model_name == "other-model"
        return np.asarray([[0.0, 1.0, 0.0]], dtype="float32")

    monkeypatch.setattr("neural.retrieval.encode_texts", fake_encode)
    results = retrieve(bundle, "q", top_k=1, model_override="other-model")
    assert results[0].chunk.source_file == "b.txt"


def test_retrieve_from_disk_loads_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_min_bundle(tmp_path)
    paths: list[Path] = []
    real_load = load_retrieval_bundle

    def tracing_load(p: Path) -> RetrievalBundle:
        paths.append(p)
        return real_load(p)

    monkeypatch.setattr("neural.retrieval.load_retrieval_bundle", tracing_load)
    monkeypatch.setattr(
        "neural.retrieval.encode_texts",
        lambda texts, model_name="fixture-model", normalize_embeddings=True, **kwargs: np.asarray(
            [[1.0, 0.0, 0.0]], dtype="float32"
        ),
    )
    retrieve_from_disk("hello", tmp_path, top_k=1)
    assert paths == [tmp_path]


def test_search_result_to_citation_shape() -> None:
    from neural.vector_index import SearchResult

    chunk = TranscriptChunk(
        episode_title="E",
        source_file="e.txt",
        start_timestamp="00:00:00",
        end_timestamp="00:00:01",
        start_seconds=0,
        end_seconds=1,
        chunk_text="hi",
        line_count=1,
    )
    result = SearchResult(rank=1, score=0.123456789, chunk=chunk)
    d = search_result_to_citation(result)
    assert d["rank"] == 1
    assert d["source_file"] == "e.txt"
    assert isinstance(d["score"], float)

    lst = citations_from_results([result])
    assert len(lst) == 1
    assert lst[0]["episode_title"] == "E"


def test_retrieve_applies_metadata_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_min_bundle(tmp_path)
    metadata_index = load_metadata_index(_write_metadata(tmp_path))
    bundle = load_retrieval_bundle(tmp_path)

    monkeypatch.setattr(
        "neural.retrieval.encode_texts",
        lambda texts, model_name="fixture-model", normalize_embeddings=True, **kwargs: np.asarray(
            [[0.0, 1.0, 0.0]],
            dtype="float32",
        ),
    )
    results = retrieve(
        bundle,
        "q",
        top_k=2,
        metadata_index=metadata_index,
        filters=RetrievalFilters(team="Celtics"),
    )
    assert len(results) == 1
    assert results[0].chunk.source_file == "b.txt"


def test_retrieve_applies_reranking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_min_bundle(tmp_path)
    bundle = load_retrieval_bundle(tmp_path)

    monkeypatch.setattr(
        "neural.retrieval.encode_texts",
        lambda texts, model_name="fixture-model", normalize_embeddings=True, **kwargs: np.asarray(
            [[1.0, 0.0, 0.0]],
            dtype="float32",
        ),
    )

    def fake_rerank(query: str, results: list, config: RerankerConfig):
        swapped = list(reversed(results))
        return [
            type(results[0])(rank=index + 1, score=result.score, chunk=result.chunk)
            for index, result in enumerate(swapped)
        ]

    monkeypatch.setattr("neural.retrieval.rerank_results", fake_rerank)
    results = retrieve(bundle, "q", top_k=2, reranker=RerankerConfig(top_n=2))
    assert results[0].chunk.source_file == "b.txt"


def test_citations_include_metadata_when_available(tmp_path: Path) -> None:
    _write_min_bundle(tmp_path)
    metadata_index = load_metadata_index(_write_metadata(tmp_path))
    bundle = load_retrieval_bundle(tmp_path)
    results = [
        SearchResult(rank=1, score=0.9, chunk=bundle.chunks[0]),
    ]
    citations = citations_from_results(results, metadata_index)
    metadata = citations[0].get("metadata")
    assert isinstance(metadata, dict)
    assert metadata["speaker"] == "Speaker A"
