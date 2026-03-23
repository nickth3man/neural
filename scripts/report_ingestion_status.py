#!/usr/bin/env python3
# ruff: noqa: E402
"""Summarize local transcript, index, and metadata artifact status."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neural.ingestion_status import summarize_ingestion_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local transcript corpus, FAISS index, and metadata artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--transcripts-dir", type=Path, default=Path("gil/transcripts"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/transcript_index"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = summarize_ingestion_status(
        transcripts_dir=args.transcripts_dir,
        index_dir=args.index_dir,
        metadata_dir=args.metadata_dir,
    )
    if args.json:
        print(json.dumps(asdict(status), indent=2))
        return
    print(f"Generated at:           {status.generated_at}")
    print(f"Transcript files:       {status.transcript_count}")
    print(f"Index present:          {status.index_present}")
    print(f"Indexed chunks:         {status.indexed_chunks}")
    print(f"Metadata present:       {status.metadata_present}")
    print(f"Enriched chunks:        {status.enriched_chunks}")
    print(f"Document metadata rows: {status.document_metadata_records}")


if __name__ == "__main__":
    main()
