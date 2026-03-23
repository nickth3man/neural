from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from neural.metadata_index import DOCUMENT_METADATA_FILENAME, ENRICHED_CHUNKS_FILENAME
from neural.vector_index import CHUNKS_FILENAME, CONFIG_FILENAME, INDEX_FILENAME


@dataclass(frozen=True, slots=True)
class IngestionStatus:
    generated_at: str
    transcript_count: int
    manifest_entries: int
    scrape_failures: int
    recent_failure_reasons: dict[str, int]
    index_present: bool
    indexed_chunks: int
    metadata_present: bool
    enriched_chunks: int
    document_metadata_records: int


def summarize_ingestion_status(
    transcripts_dir: Path,
    manifest_path: Path,
    failure_log_path: Path,
    index_dir: Path,
    metadata_dir: Path,
) -> IngestionStatus:
    transcript_count = len(list(transcripts_dir.glob("*.txt"))) if transcripts_dir.exists() else 0
    manifest_entries = _json_list_length(manifest_path)
    scrape_failures, failure_reasons = _parse_failure_log(failure_log_path)
    index_present = all(
        (index_dir / name).exists() for name in (INDEX_FILENAME, CHUNKS_FILENAME, CONFIG_FILENAME)
    )
    indexed_chunks = _json_list_length(index_dir / CHUNKS_FILENAME)
    metadata_present = all(
        (metadata_dir / name).exists()
        for name in (ENRICHED_CHUNKS_FILENAME, DOCUMENT_METADATA_FILENAME)
    )
    enriched_chunks = _json_list_length(metadata_dir / ENRICHED_CHUNKS_FILENAME)
    document_metadata_records = _json_list_length(metadata_dir / DOCUMENT_METADATA_FILENAME)
    return IngestionStatus(
        generated_at=datetime.now().isoformat(),
        transcript_count=transcript_count,
        manifest_entries=manifest_entries,
        scrape_failures=scrape_failures,
        recent_failure_reasons=failure_reasons,
        index_present=index_present,
        indexed_chunks=indexed_chunks,
        metadata_present=metadata_present,
        enriched_chunks=enriched_chunks,
        document_metadata_records=document_metadata_records,
    )


def _json_list_length(path: Path) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, list) else 0


def _parse_failure_log(path: Path) -> tuple[int, dict[str, int]]:
    if not path.exists():
        return 0, {}
    counter: Counter[str] = Counter()
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        parts = [part.strip() for part in line.split("|")]
        reason = parts[-1] if parts else "unknown"
        counter[reason] += 1
    return total, dict(counter.most_common(5))
