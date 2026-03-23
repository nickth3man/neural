"""Tests for full and incremental index builds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural.chunking import ChunkingConfig
from neural.index_build import build_index_full, build_index_incremental
from neural.vector_index import load_index_bundle

faiss = pytest.importorskip("faiss")


def _synthetic_episode_lines(n: int) -> str:
    lines = []
    for i in range(n):
        sec = 10 + i
        lines.append(
            f"Starting point is 00:00:{sec:02d}"
            f"This is sentence number {i} with enough words to satisfy minimum chunk length rules."
        )
    return "\n".join(lines)


def test_incremental_adds_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcripts = tmp_path / "tr"
    transcripts.mkdir()
    (transcripts / "one.txt").write_text(_synthetic_episode_lines(8), encoding="utf-8")

    out = tmp_path / "idx"
    cfg = ChunkingConfig(lines_per_chunk=3, line_overlap=1, min_chunk_characters=20)

    def fake_encode(texts: list[str], model_name: str = "", **_: object) -> np.ndarray:
        n = len(texts)
        dim = 8
        mat = np.zeros((n, dim), dtype=np.float32)
        for i, t in enumerate(texts):
            mat[i, i % dim] = 1.0
        faiss_mod = pytest.importorskip("faiss")
        faiss_mod.normalize_L2(mat)
        return mat

    monkeypatch.setattr("neural.index_build.encode_texts", fake_encode)

    build_index_full(
        transcripts_dir=transcripts,
        output_dir=out,
        model="fixture-model",
        chunking_config=cfg,
        limit=None,
    )
    index, chunks_after_first, _ = load_index_bundle(out)
    n_first = index.ntotal

    (transcripts / "two.txt").write_text(_synthetic_episode_lines(8), encoding="utf-8")

    assert (
        build_index_incremental(
            transcripts_dir=transcripts,
            output_dir=out,
            model="fixture-model",
            chunking_config=cfg,
        )
        is True
    )

    index2, chunks_after, _ = load_index_bundle(out)
    assert index2.ntotal > n_first
    source_files = {c.source_file for c in chunks_after}
    assert "one.txt" in source_files
    assert "two.txt" in source_files


def test_incremental_noop_when_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcripts = tmp_path / "tr"
    transcripts.mkdir()
    (transcripts / "one.txt").write_text(_synthetic_episode_lines(8), encoding="utf-8")
    out = tmp_path / "idx"
    cfg = ChunkingConfig(lines_per_chunk=3, line_overlap=1, min_chunk_characters=20)

    def fake_encode(texts: list[str], model_name: str = "", **_: object) -> np.ndarray:
        n = len(texts)
        dim = 4
        mat = np.eye(dim, dtype=np.float32)[:n]
        if mat.shape[0] < n:
            mat = np.vstack([mat, np.eye(dim, dtype=np.float32)[: n - mat.shape[0]]])
        faiss_mod = pytest.importorskip("faiss")
        faiss_mod.normalize_L2(mat)
        return mat

    monkeypatch.setattr("neural.index_build.encode_texts", fake_encode)

    build_index_full(
        transcripts_dir=transcripts,
        output_dir=out,
        model="fixture-model",
        chunking_config=cfg,
    )
    assert (
        build_index_incremental(
            transcripts_dir=transcripts,
            output_dir=out,
            model="fixture-model",
            chunking_config=cfg,
        )
        is False
    )
