from __future__ import annotations

import json
from pathlib import Path

from neural.ingestion_status import summarize_ingestion_status


def test_summarize_ingestion_status_counts_artifacts(tmp_path: Path) -> None:
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    (transcripts_dir / "one.txt").write_text("hello", encoding="utf-8")
    (transcripts_dir / "two.txt").write_text("hello", encoding="utf-8")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"file_path": "one.txt"}]), encoding="utf-8")

    failure_log = tmp_path / "failures.log"
    failure_log.write_text(
        "2026-01-01 | /foo | timeout\n2026-01-02 | /bar | timeout\n", encoding="utf-8"
    )

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    for name in ("index.faiss", "config.json"):
        (index_dir / name).write_text("x", encoding="utf-8")
    (index_dir / "chunks.json").write_text(
        json.dumps([{"chunk": 1}, {"chunk": 2}]), encoding="utf-8"
    )

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "enriched_chunks.json").write_text(json.dumps([{"chunk": 1}]), encoding="utf-8")
    (metadata_dir / "document_metadata.json").write_text(json.dumps([{"doc": 1}]), encoding="utf-8")

    status = summarize_ingestion_status(
        transcripts_dir=transcripts_dir,
        manifest_path=manifest,
        failure_log_path=failure_log,
        index_dir=index_dir,
        metadata_dir=metadata_dir,
    )
    assert status.transcript_count == 2
    assert status.manifest_entries == 1
    assert status.scrape_failures == 2
    assert status.index_present is True
    assert status.indexed_chunks == 2
    assert status.metadata_present is True
    assert status.recent_failure_reasons["timeout"] == 2
