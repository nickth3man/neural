#!/usr/bin/env python3
# ruff: noqa: E402

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
        description="Summarize scraper, index, and metadata ingestion status.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--transcripts-dir", type=Path, default=Path("gil/transcripts"))
    parser.add_argument("--manifest", type=Path, default=Path("gil/transcripts/manifest.json"))
    parser.add_argument("--failure-log", type=Path, default=Path("data/scrape_failures.log"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/transcript_index"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = summarize_ingestion_status(
        transcripts_dir=args.transcripts_dir,
        manifest_path=args.manifest,
        failure_log_path=args.failure_log,
        index_dir=args.index_dir,
        metadata_dir=args.metadata_dir,
    )
    if args.json:
        print(json.dumps(asdict(status), indent=2))
        return

    print(f"Generated:              {status.generated_at}")
    print(f"Transcript files:       {status.transcript_count}")
    print(f"Manifest entries:       {status.manifest_entries}")
    print(f"Scrape failures:        {status.scrape_failures}")
    print(f"Index present:          {status.index_present}")
    print(f"Indexed chunks:         {status.indexed_chunks}")
    print(f"Metadata present:       {status.metadata_present}")
    print(f"Enriched chunks:        {status.enriched_chunks}")
    print(f"Document metadata rows: {status.document_metadata_records}")
    if status.recent_failure_reasons:
        print("Top failure reasons:")
        for reason, count in status.recent_failure_reasons.items():
            print(f"- {reason}: {count}")


if __name__ == "__main__":
    main()
