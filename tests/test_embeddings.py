"""Tests for embedding helpers (mocked sentence-transformers)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import neural.embeddings as embeddings_mod
from neural.embeddings import encode_texts, load_embedding_model


def test_encode_texts_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        encode_texts([])


@patch("neural.embeddings.load_embedding_model")
def test_encode_texts_uses_model(mock_load: MagicMock) -> None:
    embeddings_mod.load_embedding_model.cache_clear()

    model = MagicMock()
    model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    mock_load.return_value = model

    out = encode_texts(["a", "b"], normalize_embeddings=False)

    assert out.shape == (2, 2)
    assert out.dtype == np.dtype("float32")
    model.encode.assert_called_once()
    _, kwargs = model.encode.call_args
    assert kwargs["normalize_embeddings"] is False
    embeddings_mod.load_embedding_model.cache_clear()


def test_load_embedding_model_import_error() -> None:
    embeddings_mod.load_embedding_model.cache_clear()
    with patch.object(importlib, "import_module", side_effect=ImportError("no st")):
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            load_embedding_model("m")
    embeddings_mod.load_embedding_model.cache_clear()
