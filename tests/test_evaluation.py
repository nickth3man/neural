from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural.chunking import ChunkingConfig, TranscriptChunk
from neural.evaluation import EvalCase, evaluate_retrieval, load_eval_cases
from neural.metadata_index import load_metadata_index
from neural.retrieval import load_retrieval_bundle
from neural.vector_index import build_faiss_index, save_index_artifacts

faiss = pytest.importorskip("faiss")


def _normalized(rows: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def _write_bundle(tmp_path: Path) -> None:
    embeddings = _normalized([[1.0, 0.0], [0.0, 1.0]])
    index = build_faiss_index(embeddings)
    chunks = [
        TranscriptChunk(
            episode_title="Episode_A",
            source_file="episode_a.txt",
            start_timestamp="00:00:01",
            end_timestamp="00:00:05",
            start_seconds=1,
            end_seconds=5,
            chunk_text="alpha talk",
            line_count=2,
        ),
        TranscriptChunk(
            episode_title="Episode_B",
            source_file="episode_b.txt",
            start_timestamp="00:00:10",
            end_timestamp="00:00:14",
            start_seconds=10,
            end_seconds=14,
            chunk_text="beta talk",
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
                    "source_file": "episode_a.txt",
                    "start_timestamp": "00:00:01",
                    "end_timestamp": "00:00:05",
                    "episode_type": "reaction",
                    "guest_names": ["Guest A"],
                    "host_names": ["Host A"],
                    "episode_topic": "alpha",
                    "chunk_meta": {
                        "mentioned_people": ["Person A"],
                        "mentioned_teams": ["Lakers"],
                        "mentioned_leagues": ["NBA"],
                    },
                    "llm_meta": {
                        "speaker": "Speaker A",
                        "topic": "alpha",
                        "subtopic": "trades",
                    },
                },
                {
                    "source_file": "episode_b.txt",
                    "start_timestamp": "00:00:10",
                    "end_timestamp": "00:00:14",
                    "episode_type": "guest_interview",
                    "guest_names": ["Guest B"],
                    "host_names": ["Host B"],
                    "episode_topic": "beta",
                    "chunk_meta": {
                        "mentioned_people": ["Person B"],
                        "mentioned_teams": ["Celtics"],
                        "mentioned_leagues": ["NBA"],
                    },
                    "llm_meta": {
                        "speaker": "Speaker B",
                        "topic": "beta",
                        "subtopic": "playoffs",
                    },
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "document_metadata.json").write_text(
        json.dumps(
            [
                {
                    "source_file": "episode_a.txt",
                    "episode_type": "reaction",
                    "guest_names": ["Guest A"],
                    "host_names": ["Host A"],
                    "topic": "alpha",
                    "holiday_theme": "",
                },
                {
                    "source_file": "episode_b.txt",
                    "episode_type": "guest_interview",
                    "guest_names": ["Guest B"],
                    "host_names": ["Host B"],
                    "topic": "beta",
                    "holiday_theme": "",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_dir


def test_load_eval_cases(tmp_path: Path) -> None:
    evals = tmp_path / "evals.json"
    evals.write_text(
        json.dumps(
            [{"query": "alpha", "expected_episode_substring": "episode_a", "notes": "seed"}]
        ),
        encoding="utf-8",
    )
    cases = load_eval_cases(evals)
    assert cases[0].query == "alpha"
    assert cases[0].notes == "seed"


def test_evaluate_retrieval_reports_hit_rate_and_mrr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_bundle(tmp_path)
    bundle = load_retrieval_bundle(tmp_path)

    def fake_encode(
        texts: list[str],
        *,
        model_name: str = "fixture-model",
        normalize_embeddings: bool = True,
        **_: object,
    ) -> np.ndarray:
        if texts == ["find alpha"]:
            return np.asarray([[1.0, 0.0]], dtype="float32")
        return np.asarray([[0.0, 1.0]], dtype="float32")

    monkeypatch.setattr("neural.retrieval.encode_texts", fake_encode)
    summary = evaluate_retrieval(bundle, [])
    assert summary.total_queries == 0

    cases = [
        EvalCase(query="find alpha", expected_episode_substring="episode_a"),
        EvalCase(query="find beta", expected_episode_substring="episode_b"),
    ]
    summary = evaluate_retrieval(bundle, cases, top_k=2)
    assert summary.hit_rate_at_k == 1.0
    assert summary.mrr == 1.0


def test_evaluate_retrieval_with_metadata_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_bundle(tmp_path)
    metadata_dir = _write_metadata(tmp_path)
    metadata_index = load_metadata_index(metadata_dir)
    bundle = load_retrieval_bundle(tmp_path)

    monkeypatch.setattr(
        "neural.retrieval.encode_texts",
        lambda texts, model_name="fixture-model", normalize_embeddings=True, **kwargs: np.asarray(
            [[0.0, 1.0]],
            dtype="float32",
        ),
    )
    cases = [EvalCase(query="anything", expected_episode_substring="episode_b")]
    from neural.metadata_index import RetrievalFilters

    summary = evaluate_retrieval(
        bundle,
        cases,
        top_k=2,
        metadata_index=metadata_index,
        filters=RetrievalFilters(team="Celtics"),
    )
    assert summary.hit_rate_at_k == 1.0
