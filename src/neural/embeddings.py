"""Embedding helpers for transcript retrieval (OpenRouter API only)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from neural.openrouter import OpenRouterError, embed_texts_openrouter

DEFAULT_OPENROUTER_EMBEDDING_MODEL = "mistralai/mistral-embed-2312"

EMBEDDING_BACKEND_OPENROUTER = "openrouter"

# Stored in legacy index config.json; rejected at query time after OpenRouter-only migration.
LEGACY_EMBEDDING_BACKEND_SENTENCE_TRANSFORMERS = "sentence_transformers"


def require_openrouter_embedding_model() -> str:
    """Return a non-empty OpenRouter embedding model id from ``OPENROUTER_EMBEDDING_MODEL``."""
    model = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "").strip()
    if not model:
        msg = (
            "OPENROUTER_EMBEDDING_MODEL must be set to an OpenRouter embedding model id "
            "(e.g. mistralai/mistral-embed-2312)"
        )
        raise ValueError(msg)
    return model


def default_model_for_cli() -> str:
    """Default ``--model`` for index scripts when it is omitted (from env)."""
    return require_openrouter_embedding_model()


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def encode_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    normalize_embeddings: bool = True,
    openrouter_api_key: str | None = None,
    openrouter_batch_size: int = 32,
) -> np.ndarray:
    """Encode texts to float32 embeddings via OpenRouter."""
    if not texts:
        msg = "Cannot encode an empty text sequence"
        raise ValueError(msg)

    key = (openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    if not key:
        msg = "OpenRouter embeddings require OPENROUTER_API_KEY"
        raise ValueError(msg)
    try:
        rows = embed_texts_openrouter(
            list(texts),
            model=model_name,
            api_key=key,
            batch_size=openrouter_batch_size,
        )
    except OpenRouterError:
        raise
    arr = np.asarray(rows, dtype=np.float32)
    if normalize_embeddings:
        arr = _l2_normalize_rows(arr)
    return arr
