"""Tests for OpenRouter embedding helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neural.embeddings import encode_texts, require_openrouter_embedding_model


def test_encode_texts_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        encode_texts([], model_name="mistralai/mistral-embed-2312")


def test_encode_texts_openrouter_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list[str], **_: object) -> list[list[float]]:
        return [[3.0, 4.0] for _ in texts]

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr("neural.embeddings.embed_texts_openrouter", fake_embed)

    out = encode_texts(["a", "b"], model_name="mistralai/mistral-embed-2312")
    assert out.shape == (2, 2)
    norms = np.linalg.norm(out, axis=1)
    assert norms == pytest.approx(np.array([1.0, 1.0], dtype=np.float32), rel=1e-5)


def test_encode_texts_openrouter_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        encode_texts(["x"], model_name="mistralai/mistral-embed-2312")


def test_require_openrouter_embedding_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_EMBEDDING_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_EMBEDDING_MODEL"):
        require_openrouter_embedding_model()


def test_require_openrouter_embedding_model_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "mistralai/mistral-embed-2312")
    assert require_openrouter_embedding_model() == "mistralai/mistral-embed-2312"
