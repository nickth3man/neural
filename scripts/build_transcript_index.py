#!/usr/bin/env python3
# ruff: noqa: E402
"""Build a local semantic index over Gil transcript chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural.chunking import ChunkingConfig, chunk_corpus
from neural.corpus import load_corpus
from neural.embeddings import DEFAULT_EMBEDDING_MODEL, encode_texts
from neural.vector_index import build_faiss_index, save_index_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS transcript index from local transcript files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers embedding model name",
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
        help="Optional limit on the number of transcript files to load",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunking_config = ChunkingConfig(
        lines_per_chunk=args.lines_per_chunk,
        line_overlap=args.line_overlap,
        min_chunk_characters=args.min_chars,
    )

    documents = load_corpus(args.transcripts_dir, limit=args.limit)
    chunks = chunk_corpus(documents, config=chunking_config)
    if not chunks:
        msg = "No transcript chunks were produced from the selected corpus"
        raise ValueError(msg)

    embeddings = encode_texts([chunk.chunk_text for chunk in chunks], model_name=args.model)
    index = build_faiss_index(embeddings)
    save_index_artifacts(
        output_dir=args.output_dir,
        index=index,
        chunks=chunks,
        model_name=args.model,
        chunking_config=chunking_config,
        transcripts_dir=args.transcripts_dir,
    )

    print(
        f"Indexed {len(documents)} transcripts into {len(chunks)} chunks with model {args.model}."
    )
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
