#!/usr/bin/env python3
# ruff: noqa: E402
"""Extract metadata from Gil's Arena transcripts using the glossary-first pipeline.

Pipeline:
1. Load transcript documents
2. Parse intro windows (deterministic)
3. Extract document-level metadata (deterministic + LLM)
4. Chunk transcripts
5. Enrich chunks with metadata (deterministic + LLM)
6. Save enriched chunks and document metadata

Usage:
    uv run python scripts/extract_metadata.py
    uv run python scripts/extract_metadata.py --limit 5
    uv run python scripts/extract_metadata.py --skip-llm
    uv run python scripts/extract_metadata.py --output-dir data/metadata_output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env for OPENROUTER_API_KEY and OPENROUTER_MODEL
load_dotenv(REPO_ROOT / ".env")

import os

from neural.chunk_enrichment import enrich_chunks
from neural.chunking import ChunkingConfig, chunk_transcript
from neural.corpus import load_corpus
from neural.glossary_extractor import extract_document_metadata
from neural.metadata_types import EpisodeManifestEntry

# ============================================================================
# Configuration
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract metadata from Gil's Arena transcripts.",
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
        default=Path("data/metadata"),
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.json for episode metadata",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of transcripts to process",
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
        "--skip-llm",
        action="store_true",
        help="Skip LLM extraction (deterministic only, no API key needed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


# ============================================================================
# Pipeline Functions
# ============================================================================


def load_manifest(manifest_path: Path | None) -> dict[str, EpisodeManifestEntry]:
    """Load manifest entries indexed by filename stem."""
    if not manifest_path or not manifest_path.exists():
        return {}

    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)

    manifest_map: dict[str, EpisodeManifestEntry] = {}
    for entry_data in entries:
        entry = EpisodeManifestEntry.from_dict(entry_data)
        # Index by filename stem (without .txt)
        stem = Path(entry.file_path).stem
        manifest_map[stem] = entry

    return manifest_map


def run_pipeline(
    transcripts_dir: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    limit: int | None = None,
    chunking_config: ChunkingConfig | None = None,
    skip_llm: bool = False,
    verbose: bool = False,
) -> dict:
    """Run the full metadata extraction pipeline.

    Returns:
        Dictionary with extraction statistics.
    """
    if chunking_config is None:
        chunking_config = ChunkingConfig()

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "") if not skip_llm else ""
    model = os.environ.get("OPENROUTER_MODEL", None) if not skip_llm else None

    if not skip_llm and not api_key:
        print("Warning: No OPENROUTER_API_KEY found. Falling back to deterministic extraction.")
        skip_llm = True

    # Load manifest
    manifest_map = load_manifest(manifest_path)

    # Load transcripts
    print(f"Loading transcripts from {transcripts_dir}...")
    documents = load_corpus(transcripts_dir, limit=limit)
    print(f"Loaded {len(documents)} transcripts.")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each document
    all_enriched_chunks: list[dict] = []
    all_document_metadata: list[dict] = []
    stats = {
        "total_documents": len(documents),
        "total_chunks": 0,
        "llm_extractions": 0,
        "deterministic_extractions": 0,
    }

    for i, document in enumerate(documents):
        if verbose:
            print(f"\n[{i + 1}/{len(documents)}] Processing: {document.episode_title}")

        # Get manifest entry if available
        stem = Path(document.source_file).stem
        manifest_entry = manifest_map.get(stem)

        # Step 1: Extract document-level metadata
        doc_result = extract_document_metadata(
            document,
            api_key=api_key,
            model=model,
            manifest_entry=manifest_entry,
            skip_llm=skip_llm,
        )

        if doc_result.used_llm:
            stats["llm_extractions"] += 1
        else:
            stats["deterministic_extractions"] += 1

        if verbose:
            print(f"  Episode type: {doc_result.metadata.episode_type.value}")
            print(f"  Hosts: {doc_result.metadata.host_names}")
            print(f"  Guests: {doc_result.metadata.guest_names}")
            print(f"  Roster: {doc_result.metadata.episode_roster}")
            print(f"  Holiday: {doc_result.metadata.holiday_theme or 'none'}")
            print(f"  Topic: {doc_result.metadata.topic[:80] or 'none'}")

        # Save document metadata
        doc_meta_dict = doc_result.metadata.to_dict()
        doc_meta_dict["source_file"] = document.source_file
        all_document_metadata.append(doc_meta_dict)

        # Step 2: Chunk transcript
        chunks = chunk_transcript(document, config=chunking_config)
        stats["total_chunks"] += len(chunks)

        if verbose:
            print(f"  Chunks: {len(chunks)}")

        # Step 3: Enrich chunks
        chunk_results = enrich_chunks(
            chunks,
            doc_result.metadata,
            api_key=api_key,
            model=model,
            skip_llm=skip_llm,
        )

        for chunk_result in chunk_results:
            enriched = chunk_result.enriched_chunk
            chunk_dict = enriched.to_dict()
            chunk_dict["raw_llm_response"] = chunk_result.raw_response
            chunk_dict["used_llm"] = chunk_result.used_llm
            all_enriched_chunks.append(chunk_dict)

    # Save outputs
    chunks_output = output_dir / "enriched_chunks.json"
    with open(chunks_output, "w", encoding="utf-8") as f:
        json.dump(all_enriched_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_enriched_chunks)} enriched chunks to {chunks_output}")

    metadata_output = output_dir / "document_metadata.json"
    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(all_document_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_document_metadata)} document metadata to {metadata_output}")

    # Save stats
    stats_output = output_dir / "extraction_stats.json"
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved extraction stats to {stats_output}")

    return stats


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Parse arguments and run the metadata extraction pipeline."""
    args = parse_args()

    chunking_config = ChunkingConfig(
        lines_per_chunk=args.lines_per_chunk,
        line_overlap=args.line_overlap,
        min_chunk_characters=40,
    )

    print("=" * 60)
    print("Gil's Arena Transcript Metadata Extraction")
    print("=" * 60)

    try:
        stats = run_pipeline(
            transcripts_dir=args.transcripts_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            limit=args.limit,
            chunking_config=chunking_config,
            skip_llm=args.skip_llm,
            verbose=args.verbose,
        )

        print("\n" + "=" * 60)
        print("Extraction Summary")
        print("=" * 60)
        print(f"Documents processed:     {stats['total_documents']}")
        print(f"Chunks produced:         {stats['total_chunks']}")
        print(f"LLM extractions:         {stats['llm_extractions']}")
        print(f"Deterministic extractions: {stats['deterministic_extractions']}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nExtraction interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
