#!/usr/bin/env python3
# ruff: noqa: E402
"""Query a previously built transcript index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neural.metadata_index import RetrievalFilters, load_metadata_index
from neural.reranking import RerankerConfig
from neural.retrieval import retrieve_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query a saved FAISS transcript index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", type=str, help="Free-text query to search for")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/transcript_index"),
        help="Directory containing index.faiss, chunks.json, and config.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of matches to return",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the embedding model recorded in config.json",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help=(
            "Optional metadata directory containing enriched_chunks.json and document_metadata.json"
        ),
    )
    parser.add_argument("--episode-type", type=str, default=None, help="Filter by episode type")
    parser.add_argument("--guest", type=str, default=None, help="Filter by guest name")
    parser.add_argument("--speaker", type=str, default=None, help="Filter by detected speaker")
    parser.add_argument("--team", type=str, default=None, help="Filter by mentioned team")
    parser.add_argument("--topic", type=str, default=None, help="Filter by episode or chunk topic")
    parser.add_argument("--source-file", type=str, default=None, help="Filter by source filename")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking")
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=RerankerConfig().model_name,
        help="Cross-encoder model for reranking",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=RerankerConfig().top_n,
        help="Number of retrieved candidates to rerank",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_index = load_metadata_index(args.metadata_dir) if args.metadata_dir else None
    filters = RetrievalFilters(
        episode_type=args.episode_type,
        guest_name=args.guest,
        speaker=args.speaker,
        team=args.team,
        topic=args.topic,
        source_file=args.source_file,
    )
    reranker = None
    if args.rerank:
        reranker = RerankerConfig(model_name=args.reranker_model, top_n=args.rerank_top_n)
    results = retrieve_from_disk(
        args.query,
        args.index_dir,
        top_k=args.top_k,
        model_override=args.model,
        metadata_index=metadata_index,
        filters=filters,
        reranker=reranker,
    )

    if not results:
        print("No results found.")
        return

    for result in results:
        chunk = result.chunk
        print(
            f"{result.rank}. score={result.score:.4f} "
            f"{chunk.episode_title} [{chunk.start_timestamp}-{chunk.end_timestamp}]"
        )
        print(f"   file={chunk.source_file}")
        print(f"   text={chunk.chunk_text}")
        if metadata_index is not None:
            metadata = metadata_index.metadata_for_result(result)
            if metadata is not None:
                print(
                    f"   metadata=episode_type:{metadata.episode_type or '-'} "
                    f"speaker:{metadata.speaker or '-'} "
                    f"teams:{', '.join(metadata.mentioned_teams) or '-'}"
                )


if __name__ == "__main__":
    main()
