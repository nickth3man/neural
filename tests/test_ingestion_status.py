"""Tests for local artifact ingestion status (static corpus)."""

from __future__ import annotations

import json
from pathlib import Path

from neural.ingestion_status import summarize_ingestion_status


def test_summarize_ingestion_status_counts_artifacts(tmp_path: Path) -> None:
    transcripts = tmp_path / "t"
    transcripts.mkdir()
    (transcripts / "one.txt").write_text("a", encoding="utf-8")
    (transcripts / "two.txt").write_text("b", encoding="utf-8")

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"x")
    (index_dir / "chunks.json").write_text(json.dumps([{"x": 1}, {"x": 2}]), encoding="utf-8")
    (index_dir / "config.json").write_text("{}", encoding="utf-8")

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "enriched_chunks.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (meta_dir / "document_metadata.json").write_text(json.dumps([{"id": 1}]), encoding="utf-8")

    status = summarize_ingestion_status(
        transcripts_dir=transcripts,
        index_dir=index_dir,
        metadata_dir=meta_dir,
    )
    assert status.transcript_count == 2
    assert status.index_present is True
    assert status.indexed_chunks == 2
    assert status.metadata_present is True
    assert status.enriched_chunks == 3
    assert status.document_metadata_records == 1
