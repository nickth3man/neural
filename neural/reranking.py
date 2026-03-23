from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache

from neural.vector_index import SearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    model_name: str = DEFAULT_RERANKER_MODEL
    top_n: int = 20


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str):
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        msg = (
            "sentence-transformers is required for reranking support. "
            "Install project dependencies before using reranking."
        )
        raise RuntimeError(msg) from exc
    CrossEncoder = getattr(module, "CrossEncoder")
    return CrossEncoder(model_name)


def rerank_results(
    query: str,
    results: list[SearchResult],
    config: RerankerConfig,
) -> list[SearchResult]:
    if not results:
        return []

    limited = results[: config.top_n]
    model = _load_cross_encoder(config.model_name)
    pairs = [(query, result.chunk.chunk_text) for result in limited]
    scores = model.predict(pairs, show_progress_bar=False)
    reranked_pairs = sorted(
        zip(limited, scores, strict=True), key=lambda item: float(item[1]), reverse=True
    )

    reranked: list[SearchResult] = []
    for rank, (result, score) in enumerate(reranked_pairs, start=1):
        reranked.append(SearchResult(rank=rank, score=float(score), chunk=result.chunk))

    for result in results[config.top_n :]:
        reranked.append(
            SearchResult(rank=len(reranked) + 1, score=result.score, chunk=result.chunk)
        )
    return reranked
