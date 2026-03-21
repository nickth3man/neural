"""Tests for FastAPI chatbot (webapp)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from neural.chunking import ChunkingConfig, TranscriptChunk
from neural.vector_index import build_faiss_index, save_index_artifacts

faiss = pytest.importorskip("faiss")


def _normalized(rows: list[list[float]]):
    import numpy as np

    matrix = np.asarray(rows, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def _save_index(tmp_path: Path) -> None:
    embeddings = _normalized([[1.0, 0.0]])
    index = build_faiss_index(embeddings)
    chunks = [
        TranscriptChunk(
            episode_title="Test_Ep",
            source_file="test.txt",
            start_timestamp="00:00:00",
            end_timestamp="00:00:01",
            start_seconds=0,
            end_seconds=1,
            chunk_text="evidence line",
            line_count=1,
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


@pytest.fixture
def chat_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _save_index(tmp_path)
    monkeypatch.setenv("GIL_INDEX_DIR", str(tmp_path))

    def fake_encode(
        texts: list[str],
        *,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
    ):
        import numpy as np

        return np.asarray([[1.0, 0.0]], dtype="float32")

    monkeypatch.setattr("neural.retrieval.encode_texts", fake_encode)
    import webapp.main as wm

    importlib.reload(wm)
    with TestClient(wm.app) as client:
        yield client


def test_root_ok(chat_client: TestClient) -> None:
    r = chat_client.get("/")
    assert r.status_code == 200
    assert "Gil" in r.text


def test_api_chat_retrieval_only(chat_client: TestClient) -> None:
    r = chat_client.post(
        "/api/chat",
        json={"message": "anything", "top_k": 1, "retrieval_only": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["generation_skipped"] is True
    assert len(data["citations"]) == 1
    assert "Test_Ep" in data["answer"] or "test.txt" in data["answer"]


def test_api_chat_no_key_returns_skipped_generation(
    chat_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    r = chat_client.post(
        "/api/chat",
        json={"message": "hello", "top_k": 3, "retrieval_only": False},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["generation_skipped"] is True
    assert data["error"] == "OPENROUTER_API_KEY not set"
    assert len(data["citations"]) >= 1


def test_api_chat_validation_error(chat_client: TestClient) -> None:
    r = chat_client.post("/api/chat", json={"message": "", "top_k": 0})
    assert r.status_code == 422


def test_api_chat_index_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "no_index_here"
    empty.mkdir()
    monkeypatch.setenv("GIL_INDEX_DIR", str(empty))
    import webapp.main as wm

    importlib.reload(wm)
    with TestClient(wm.app) as client:
        r = client.post("/api/chat", json={"message": "x", "top_k": 1, "retrieval_only": True})
    assert r.status_code == 503
