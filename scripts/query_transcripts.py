#!/usr/bin/env python3
# ruff: noqa: E402
"""Query a previously built transcript index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve_from_disk(
        args.query,
        args.index_dir,
        top_k=args.top_k,
        model_override=args.model,
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


if __name__ == "__main__":
    main()
