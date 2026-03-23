"""Content-addressed manifest of transcript files for incremental indexing."""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_file(path: Path) -> str:
    """Return lowercase hex SHA-256 of file contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_transcript_manifest(transcripts_dir: Path, *, pattern: str = "*.txt") -> dict[str, str]:
    """Map relative POSIX path (from ``transcripts_dir``) to content hash for each transcript."""
    if not transcripts_dir.exists():
        return {}
    manifest: dict[str, str] = {}
    for path in sorted(transcripts_dir.glob(pattern)):
        if not path.is_file():
            continue
        rel = path.relative_to(transcripts_dir).as_posix()
        manifest[rel] = hash_file(path)
    return manifest


def diff_transcript_manifest(
    previous: dict[str, str],
    current: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    """Return ``(added, removed, changed)`` paths (POSIX strings)."""
    prev_keys = set(previous)
    curr_keys = set(current)
    added = curr_keys - prev_keys
    removed = prev_keys - curr_keys
    changed = {key for key in prev_keys & curr_keys if previous[key] != current[key]}
    return added, removed, changed
