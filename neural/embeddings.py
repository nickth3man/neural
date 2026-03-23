"""Embedding helpers for transcript retrieval."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load a sentence-transformers model lazily."""
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        msg = (
            "sentence-transformers is required for embedding support. "
            "Install project dependencies before building or querying the index."
        )
        raise RuntimeError(msg) from exc

    SentenceTransformer = getattr(module, "SentenceTransformer")
    return SentenceTransformer(model_name)


def encode_texts(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Encode text into normalized float32 embeddings."""
    if not texts:
        msg = "Cannot encode an empty text sequence"
        raise ValueError(msg)

    model = load_embedding_model(model_name)
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")
