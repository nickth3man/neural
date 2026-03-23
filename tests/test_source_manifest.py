"""Tests for transcript source manifest hashing and diffs."""

from __future__ import annotations

from pathlib import Path

from neural.source_manifest import (
    compute_transcript_manifest,
    diff_transcript_manifest,
    hash_file,
)


def test_hash_file_stable(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    assert hash_file(p) == hash_file(p)


def test_diff_transcript_manifest(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("one", encoding="utf-8")
    b.write_text("two", encoding="utf-8")
    prev = {"a.txt": "x", "b.txt": "y"}
    curr = compute_transcript_manifest(tmp_path)
    added, removed, changed = diff_transcript_manifest(prev, curr)
    assert added == set()
    assert removed == set()
    assert changed == {"a.txt", "b.txt"}

    added, removed, changed = diff_transcript_manifest({}, curr)
    assert added == {"a.txt", "b.txt"}

    added, removed, changed = diff_transcript_manifest(curr, {})
    assert removed == {"a.txt", "b.txt"}
