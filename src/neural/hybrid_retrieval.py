"""Lexical retrieval and reciprocal rank fusion for hybrid search."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from neural.chunking import TranscriptChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for BM25."""
    return _TOKEN_RE.findall(text.lower())


def bm25_ranked_chunk_ids(chunks: list[TranscriptChunk], query: str, top_n: int) -> list[int]:
    """Return up to ``top_n`` chunk ids ordered by BM25 score (descending)."""
    if top_n < 1:
        msg = "top_n must be at least 1"
        raise ValueError(msg)
    if not chunks:
        return []

    tokenized_corpus = [tokenize_for_bm25(c.chunk_text) for c in chunks]
    if all(len(t) == 0 for t in tokenized_corpus):
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    out: list[int] = []
    for idx in order[:top_n]:
        chunk = chunks[int(idx)]
        if chunk.chunk_id is None:
            continue
        out.append(int(chunk.chunk_id))
    return out


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, rrf_k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ordered id lists with RRF; return ``(id, score)`` sorted by score descending."""
    if rrf_k < 1:
        msg = "rrf_k must be at least 1"
        raise ValueError(msg)
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
