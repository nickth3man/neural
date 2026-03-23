"""Tests for FastAPI chatbot (webapp)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from neural.chunking import ChunkingConfig, TranscriptChunk
from neural.vector_index import build_faiss_index, save_index_artifacts

faiss = pytest.importorskip("faiss")


def _parse_sse_events(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[7:].strip()
            elif line.startswith("data: "):
                data_lines.append(line[6:])
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


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


def test_health_endpoint(chat_client: TestClient) -> None:
    r = chat_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["index_loaded"] is True
    assert "last_chat_ms" in data["metrics"]
    assert "last_stream_ms" in data["metrics"]


def test_api_metrics_endpoint(chat_client: TestClient) -> None:
    r = chat_client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "started_at" in data
    assert "uptime_seconds" in data
    assert "metrics" in data
    assert "request_count" in data["metrics"]


def test_ready_endpoint_requires_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "no_index_here"
    empty.mkdir()
    monkeypatch.setenv("GIL_INDEX_DIR", str(empty))
    import webapp.main as wm

    importlib.reload(wm)
    with TestClient(wm.app) as client:
        r = client.get("/ready")
    assert r.status_code == 503


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
    assert data.get("answer_html")
    assert "Test_Ep" in data["answer_html"] or "test.txt" in data["answer_html"]


def test_api_chat_history_passed_to_complete_chat(
    chat_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    captured: list[list[dict[str, str]]] = []

    def fake_complete(
        messages: list[dict[str, str]],
        *,
        api_key: str,
        model: str | None = None,
    ) -> str:
        captured.append(messages)
        return "final"

    monkeypatch.setattr("webapp.main.complete_chat", fake_complete)
    r = chat_client.post(
        "/api/chat",
        json={
            "message": "latest",
            "top_k": 1,
            "retrieval_only": False,
            "history": [
                {"role": "user", "content": "prior question"},
                {"role": "assistant", "content": "prior answer"},
            ],
        },
    )
    assert r.status_code == 200
    assert captured
    contents = [m.get("content", "") for m in captured[0]]
    assert any("prior question" in c for c in contents)
    assert any("prior answer" in c for c in contents)


def test_api_chat_history_over_max_rejected(chat_client: TestClient) -> None:
    history = [{"role": "user", "content": f"t{i}"} for i in range(41)]
    r = chat_client.post(
        "/api/chat",
        json={"message": "q", "top_k": 1, "retrieval_only": True, "history": history},
    )
    assert r.status_code == 422


def test_api_chat_history_empty_content_rejected(chat_client: TestClient) -> None:
    r = chat_client.post(
        "/api/chat",
        json={
            "message": "q",
            "top_k": 1,
            "retrieval_only": True,
            "history": [{"role": "user", "content": ""}],
        },
    )
    assert r.status_code == 422


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


def test_api_chat_stream_retrieval_only(chat_client: TestClient) -> None:
    with chat_client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "anything", "top_k": 1, "retrieval_only": True},
    ) as response:
        text = "".join(response.iter_text())
    assert response.status_code == 200
    assert "event: citations" in text
    assert "event: done" in text
    events = _parse_sse_events(text)
    names = [e[0] for e in events]
    assert names.index("citations") < names.index("answer")
    assert names.index("answer") < names.index("done")
    done_payload = next(p for name, p in events if name == "done")
    assert done_payload.get("answer_html")


def test_api_chat_stream_generation(
    chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        "webapp.main.stream_chat", lambda messages, api_key, model=None: iter(["Hello", " world"])
    )
    with chat_client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "anything", "top_k": 1, "retrieval_only": False},
    ) as response:
        text = "".join(response.iter_text())
    assert response.status_code == 200
    assert "Hello" in text
    events = _parse_sse_events(text)
    done_payload = next(p for name, p in events if name == "done")
    assert "Hello world" in done_payload.get("answer", "")
    assert done_payload.get("answer_html")
