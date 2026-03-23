#!/usr/bin/env python3
# ruff: noqa: E402
"""Cluster transcript chunks in embedding space and write a JSON topic report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(REPO_ROOT / ".env")

from neural.cluster_topics import kmeans_cosine
from neural.embeddings import (
    EMBEDDING_BACKEND_OPENROUTER,
    LEGACY_EMBEDDING_BACKEND_SENTENCE_TRANSFORMERS,
    encode_texts,
)
from neural.hybrid_retrieval import tokenize_for_bm25
from neural.vector_index import load_index_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster indexed transcript chunks and write topics_report.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data/transcript_index"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/topics_report.json"),
        help="Where to write the JSON report",
    )
    parser.add_argument("--clusters", type=int, default=12, help="Number of k-means clusters")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for centroid init")
    return parser.parse_args()


def _top_tokens(texts: list[str], *, limit: int = 8) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(tokenize_for_bm25(t))
    for stop in ("the", "and", "for", "that", "this", "with", "you", "are", "was", "have"):
        counts.pop(stop, None)
    return counts.most_common(limit)


def main() -> None:
    args = parse_args()
    _, chunks, config = load_index_bundle(args.index_dir)
    model_name = str(config["model_name"])
    embedding_backend = config.get(
        "embedding_backend",
        LEGACY_EMBEDDING_BACKEND_SENTENCE_TRANSFORMERS,
    )
    if embedding_backend != EMBEDDING_BACKEND_OPENROUTER:
        msg = (
            "Clustering requires an index built with OpenRouter embeddings. "
            "Rebuild the index with OPENROUTER_EMBEDDING_MODEL and OPENROUTER_API_KEY."
        )
        raise ValueError(msg)
    texts = [c.chunk_text for c in chunks]
    embeddings = encode_texts(texts, model_name=model_name)

    labels, _ = kmeans_cosine(embeddings, args.clusters, seed=args.seed)
    by_cluster: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels.tolist()):
        by_cluster.setdefault(int(lab), []).append(idx)

    clusters_out: list[dict[str, object]] = []
    for cid in sorted(by_cluster):
        member_idx = by_cluster[cid]
        member_chunks = [chunks[i] for i in member_idx]
        sample_titles = [c.episode_title for c in member_chunks[:5]]
        top_words = _top_tokens([c.chunk_text for c in member_chunks])
        clusters_out.append(
            {
                "cluster_id": cid,
                "size": len(member_idx),
                "sample_episode_titles": sample_titles,
                "top_tokens": [w for w, _ in top_words],
            },
        )

    report = {
        "index_dir": str(args.index_dir.resolve()),
        "model_name": model_name,
        "embedding_backend": embedding_backend,
        "total_chunks": len(chunks),
        "k": args.clusters,
        "clusters": clusters_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(clusters_out)} clusters from {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
