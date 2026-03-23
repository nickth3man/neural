"""Full and incremental FAISS transcript index builds (library entry points)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from neural.chunking import ChunkingConfig, chunk_corpus, chunk_transcript
from neural.corpus import load_corpus, load_transcript
from neural.embeddings import EMBEDDING_BACKEND_OPENROUTER, encode_texts
from neural.source_manifest import compute_transcript_manifest, diff_transcript_manifest
from neural.vector_index import (
    _is_idmap_index,
    add_vectors_with_ids,
    build_faiss_index,
    load_index_bundle,
    load_source_manifest,
    remove_ids_from_index,
    save_index_artifacts,
)


def _config_matches(config: dict, model: str, chunking: ChunkingConfig) -> bool:
    if config.get("model_name") != model:
        return False
    if config.get("embedding_backend") != EMBEDDING_BACKEND_OPENROUTER:
        return False
    return config.get("chunking") == chunking.to_dict()


def build_index_full(
    *,
    transcripts_dir: Path,
    output_dir: Path,
    model: str,
    chunking_config: ChunkingConfig,
    limit: int | None = None,
) -> tuple[int, int]:
    """Rebuild the index from scratch and write ``source_manifest.json``.

    Returns:
        ``(transcript_count, chunk_count)``.
    """
    documents = load_corpus(transcripts_dir, limit=limit)
    chunks = chunk_corpus(documents, config=chunking_config)
    if not chunks:
        msg = "No transcript chunks were produced from the selected corpus"
        raise ValueError(msg)

    embeddings = encode_texts([chunk.chunk_text for chunk in chunks], model_name=model)
    index = build_faiss_index(embeddings)
    manifest = compute_transcript_manifest(transcripts_dir)
    if limit is not None:
        allowed = {Path(d.source_file).as_posix() for d in documents}
        manifest = {k: v for k, v in manifest.items() if k in allowed}

    save_index_artifacts(
        output_dir=output_dir,
        index=index,
        chunks=chunks,
        model_name=model,
        chunking_config=chunking_config,
        transcripts_dir=transcripts_dir,
        source_manifest=manifest,
        embedding_backend=EMBEDDING_BACKEND_OPENROUTER,
    )
    return len(documents), len(chunks)


def build_index_incremental(
    *,
    transcripts_dir: Path,
    output_dir: Path,
    model: str,
    chunking_config: ChunkingConfig,
) -> bool:
    """Apply manifest diff to an existing ID-mapped index. Returns False if no work was needed."""
    index, chunks, config = load_index_bundle(output_dir)
    if not _is_idmap_index(index):
        msg = (
            "Incremental update requires an IndexIDMap2 index. "
            "Run one full build to migrate, then use incremental mode."
        )
        raise ValueError(msg)
    if not _config_matches(config, model, chunking_config):
        msg = "Index model, embedding backend, or chunking config differs from CLI; run a full rebuild"
        raise ValueError(msg)

    prev_manifest = load_source_manifest(output_dir)
    curr_manifest = compute_transcript_manifest(transcripts_dir)
    if not prev_manifest and index.ntotal > 0:
        msg = (
            "Missing source_manifest.json on a non-empty index. "
            "Run one full build to create it before incremental updates."
        )
        raise ValueError(msg)

    added, removed, changed = diff_transcript_manifest(prev_manifest, curr_manifest)
    if not added and not removed and not changed:
        return False

    touched = removed | changed
    ids_to_remove = sorted(
        {c.chunk_id for c in chunks if c.chunk_id is not None and c.source_file in touched}
    )
    if ids_to_remove:
        remove_ids_from_index(index, np.array(ids_to_remove, dtype=np.int64))

    kept = [c for c in chunks if c.source_file not in touched]

    new_docs = []
    for rel in sorted(added | changed):
        path = transcripts_dir / rel
        if not path.is_file():
            msg = f"Expected transcript file missing: {path}"
            raise FileNotFoundError(msg)
        new_docs.append(load_transcript(path))

    next_id = max((c.chunk_id for c in kept if c.chunk_id is not None), default=-1) + 1
    new_chunk_rows = []
    new_ids_list: list[int] = []
    for document in new_docs:
        for raw in chunk_transcript(document, config=chunking_config):
            new_chunk_rows.append(replace(raw, chunk_id=next_id))
            new_ids_list.append(next_id)
            next_id += 1

    if new_chunk_rows:
        embeddings = encode_texts([c.chunk_text for c in new_chunk_rows], model_name=model)
        add_vectors_with_ids(
            index,
            embeddings,
            np.array(new_ids_list, dtype=np.int64),
        )

    merged_chunks = kept + new_chunk_rows
    merged_chunks.sort(key=lambda c: c.chunk_id if c.chunk_id is not None else 0)

    save_index_artifacts(
        output_dir=output_dir,
        index=index,
        chunks=merged_chunks,
        model_name=model,
        chunking_config=chunking_config,
        transcripts_dir=transcripts_dir,
        source_manifest=curr_manifest,
        embedding_backend=EMBEDDING_BACKEND_OPENROUTER,
    )
    return True
