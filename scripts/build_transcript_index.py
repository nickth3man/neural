#!/usr/bin/env python3
# ruff: noqa: E402
"""Build a local semantic index over Gil transcript chunks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(REPO_ROOT / ".env")

from neural.chunking import ChunkingConfig
from neural.embeddings import require_openrouter_embedding_model
from neural.index_build import build_index_full, build_index_incremental


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS transcript index from local transcript files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Zero-downtime swap: build to a separate directory (e.g. data/transcript_index.new) "
            "then replace the live directory when the app can reload, or stop the app, swap "
            "folders, and restart."
        ),
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("gil/transcripts"),
        help="Directory containing transcript .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/transcript_index"),
        help="Directory where the FAISS index and metadata will be written",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "OpenRouter embedding model id (overrides OPENROUTER_EMBEDDING_MODEL; "
            "e.g. mistralai/mistral-embed-2312)"
        ),
    )
    parser.add_argument(
        "--lines-per-chunk",
        type=int,
        default=5,
        help="Number of transcript lines per chunk",
    )
    parser.add_argument(
        "--line-overlap",
        type=int,
        default=1,
        help="Number of overlapping lines between adjacent chunks",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Minimum normalized chunk text length",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of transcript files to load (full build only)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of transcript files to skip before loading (for sequential batching)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Update an existing IndexIDMap2 index using source_manifest.json (run full build once first)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunking_config = ChunkingConfig(
        lines_per_chunk=args.lines_per_chunk,
        line_overlap=args.line_overlap,
        min_chunk_characters=args.min_chars,
    )

    require_openrouter_embedding_model()
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        msg = "OPENROUTER_API_KEY is required to build or update the index"
        raise ValueError(msg)

    model = (args.model or "").strip() or require_openrouter_embedding_model()

    if args.incremental:
        if args.limit is not None:
            msg = "--limit cannot be used with --incremental"
            raise ValueError(msg)
        did_work = build_index_incremental(
            transcripts_dir=args.transcripts_dir,
            output_dir=args.output_dir,
            model=model,
            chunking_config=chunking_config,
        )
        if not did_work:
            print("Incremental: manifest unchanged; nothing to do.")
        else:
            print(f"Incremental index updated at {args.output_dir}")
        return

    n_docs, n_chunks = build_index_full(
        transcripts_dir=args.transcripts_dir,
        output_dir=args.output_dir,
        model=model,
        chunking_config=chunking_config,
        limit=args.limit,
        offset=args.offset,
    )
    print(
        f"Indexed {n_docs} transcripts into {n_chunks} chunks with model {model}.",
    )
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
