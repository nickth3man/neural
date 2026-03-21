# Gil Transcript Retrieval Tech Spec (Implementation)

## 1. Scope And Deliverable

This document specifies the retrieval MVP for the Gil transcript corpus.

The implementation must ship:

- a corpus loader for transcript files in `gil/transcripts`,
- a line-window chunker that preserves timestamp provenance,
- an embedding layer using sentence-transformers,
- a FAISS exact-search index over normalized embeddings,
- an index-build CLI,
- and a query CLI that prints top-k chunk matches.

The default local workflow is:

1. `scripts/build_transcript_index.py` reads transcript files and writes local index artifacts.
2. `scripts/query_transcripts.py` embeds a user query and prints ranked chunk matches.

## 2. Corpus Parsing Rules

### Input Assumptions

- Source files are UTF-8 text files under `gil/transcripts`.
- The dominant line format is `Starting point is HH:MM:SS<text>`.
- Blank lines are non-content and must be skipped.

### Loader Rules

1. Trim line endings and ignore empty lines.
2. If a line matches `^Starting point is (?P<timestamp>\d{2}:\d{2}:\d{2})(?P<text>.*)$`:
   - parse the timestamp into both total seconds and canonical `HH:MM:SS` form,
   - normalize internal whitespace in the remaining text to single spaces,
   - and emit one structured transcript line.
3. If a non-empty line does not match the timestamp pattern and a prior structured line exists:
   - treat the text as a continuation line,
   - normalize whitespace,
   - and append it to the prior line with a single separating space.
4. If a non-empty line does not match the timestamp pattern and no prior structured line exists:
   - skip the line,
   - count it as `skipped_prefix_lines`,
   - and continue without failing the file load.
5. Preserve source provenance on every loaded document:
   - `episode_title` from `Path.stem`,
   - `source_file` from the filename,
   - `source_path` as a string path.

### Timestamp Formatting Rules

- Internal ordering uses integer seconds.
- Displayed timestamps use zero-padded `HH:MM:SS`.
- `start_timestamp` for a chunk is the first line timestamp in that chunk.
- `end_timestamp` for a chunk is the last line timestamp in that chunk.

## 3. Chunking Rules

The retrieval MVP uses deterministic line-window chunking.

### Default Configuration

| Setting | Value |
|---------|-------|
| `lines_per_chunk` | `5` |
| `line_overlap` | `1` |
| `min_chunk_characters` | `40` |

### Chunking Algorithm

1. Build chunks only from structured timestamped transcript lines.
2. Slide a fixed window of `lines_per_chunk` across transcript lines.
3. Advance by `lines_per_chunk - line_overlap`.
4. Join chunk text by single spaces in original order.
5. Skip any chunk whose normalized text is shorter than `min_chunk_characters`.
6. If the final remainder contains at least 2 lines and meets the character minimum, emit a final partial chunk.
7. Each chunk record must include:
   - `episode_title`
   - `source_file`
   - `start_timestamp`
   - `end_timestamp`
   - `start_seconds`
   - `end_seconds`
   - `chunk_text`
   - `line_count`

## 4. Embeddings And Indexing

### Embedding Rules

| Setting | Value |
|---------|-------|
| Default model | `all-MiniLM-L6-v2` |
| Secondary candidate | `multi-qa-mpnet-base-cos-v1` |
| Normalization | `True` |
| Output dtype | `float32` |

The embedding module must:

- load a sentence-transformers model by name,
- encode chunk text and query text,
- normalize embeddings before search,
- and return `numpy.float32` arrays.

### Indexing Rules

| Setting | Value |
|---------|-------|
| FAISS index type | `IndexFlatIP` |
| Similarity mode | Inner product on normalized vectors |
| Metadata store | JSON chunk metadata |

The index module must:

- validate that embeddings are 2D and non-empty before indexing,
- build an exact-search FAISS index,
- save the index to disk,
- save chunk metadata alongside the index,
- and support query-time top-k search.

### Baseline Verification Rule

For the test fixture set, the FAISS top-k results must match the top-k order from a pure in-memory cosine similarity calculation over the same normalized embeddings.

## 5. CLI And Artifact Surface

### Build Script

`scripts/build_transcript_index.py` must accept at least:

- `--transcripts-dir`
- `--output-dir`
- `--model`
- `--lines-per-chunk`
- `--line-overlap`
- `--min-chars`
- `--limit`

The script writes:

- `index.faiss`
- `chunks.json`
- `config.json`

Default output directory: `data/transcript_index`

### Query Script

`scripts/query_transcripts.py` must accept:

- a positional query string,
- `--index-dir`
- `--top-k`
- `--model`

If `--model` is omitted, the script loads the model recorded in `config.json`.

The query output must print:

- rank,
- similarity score,
- episode title,
- source file,
- start and end timestamps,
- chunk text.

## 6. Anti-Patterns (DO NOT)

| ❌ Don't | ✅ Do Instead | Why |
|----------|---------------|-----|
| Treat full episodes as the first retrieval unit | Chunk by timestamped line windows | Full-episode retrieval hides evidence and hurts precision |
| Use `IndexFlatL2` with unnormalized embeddings for cosine-like search | Use normalized embeddings with `IndexFlatIP` | Ranking would be inconsistent with cosine retrieval expectations |
| Hardcode transcript paths in multiple modules | Centralize path handling in scripts and loader arguments | Path drift breaks reproducibility and tests |
| Fail the whole corpus load on one malformed line | Skip or merge malformed untimed lines by rule | The local corpus is semi-structured and must degrade gracefully |
| Download a model during unit tests | Use synthetic embeddings for index tests | Tests must stay fast and deterministic |
| Add ANN complexity before a verified exact baseline | Start with exact FAISS search | Correctness is more important than premature optimization |

## 7. Test Case Specifications

### Unit Tests
| Test ID | Component | Input | Expected Output | Edge Cases |
|---------|-----------|-------|-----------------|------------|
| `TC-001` | Timestamp parser | `00:01:11` | `71` seconds | `00:00:00`, `23:59:59` |
| `TC-002` | Transcript line parser | `Starting point is 00:00:24Hello world` | timestamp `00:00:24`, text `Hello world` | extra spaces, blank suffix |
| `TC-003` | Corpus loader continuation handling | one timestamped line followed by untimed text | untimed text merged into previous structured line | leading untimed line skipped |
| `TC-004` | Chunker overlap | 6 structured lines with `lines_per_chunk=3`, `line_overlap=1` | 3 chunks with overlapping middle line | final partial chunk |
| `TC-005` | FAISS search ordering | normalized fixture embeddings + query vector | same ranking as in-memory cosine search | `top_k=1`, `top_k>matches` |

### Integration Tests
| Test ID | Flow | Setup | Verification | Teardown |
|---------|------|-------|--------------|----------|
| `IT-001` | Build index from fixture transcripts | create temp transcript directory and run build helpers | `index.faiss`, `chunks.json`, and `config.json` are written | remove temp directory |
| `IT-002` | Query saved index | build temp index from fixture embeddings and metadata | query returns ranked chunk results with timestamps | remove temp directory |
| `IT-003` | Config-driven model resolution | write `config.json` with a model name and call config loader | query path uses config model when flag omitted | remove temp directory |

## 8. Error Handling Matrix

### File And Corpus Errors
| Error Type | Detection | Response | Fallback | Logging |
|------------|-----------|----------|----------|---------|
| Transcript directory missing | path does not exist | raise `FileNotFoundError` | none | `ERROR` |
| No transcript files found | directory scan returns zero matches | raise `ValueError` | none | `ERROR` |
| Leading untimed lines | non-empty unmatched line before first timestamp | skip line and increment counter | continue loading file | `WARNING` |
| Malformed chunking config | overlap >= chunk size or chunk size < 1 | raise `ValueError` | none | `ERROR` |

### Index And Query Errors
| Error Type | Detection | Response | Fallback | Logging |
|------------|-----------|----------|----------|---------|
| Empty embeddings matrix | zero rows before index build | raise `ValueError` | none | `ERROR` |
| FAISS unavailable | import failure for `faiss` | raise informative `RuntimeError` | none | `ERROR` |
| Missing index artifacts | absent `index.faiss`, `chunks.json`, or `config.json` | raise `FileNotFoundError` | none | `ERROR` |
| Query top-k invalid | `top_k < 1` | raise `ValueError` | none | `ERROR` |

## 9. Chat Application Surface (Phase Two)

The optional local web chatbot (`webapp/main.py`) must:

- Load the same index bundle as the query CLI (`index.faiss`, `chunks.json`, `config.json`).
- Call shared retrieval logic (`neural.retrieval.retrieve`) so ranking matches `scripts/query_transcripts.py`.
- Expose citations for every hit: `rank`, `score`, `episode_title`, `source_file`, `start_timestamp`, `end_timestamp`, `chunk_text`.
- Support **retrieval-only** responses (no LLM) when requested or when generation is unavailable.
- Use OpenRouter only for text generation; **do not** replace local query embeddings with a remote embedder without rebuilding the index.

### API Shape (informal)

| Concern | Rule |
|---------|------|
| `POST /api/chat` | Body: `message`, optional `top_k`, `retrieval_only`, optional `history` (short list of `{role, content}`). Response: `answer`, `citations` (list of chunk records), `generation_skipped` (bool), optional `error`. |
| `GET /` | Serves the HTML chat UI. |

### Error Handling (chat)

| Error Type | Detection | Response |
|------------|-----------|----------|
| Missing index artifacts | `FileNotFoundError` from bundle load | HTTP 503 with message to build index |
| Invalid `top_k` | `top_k < 1` | HTTP 400 |
| OpenRouter failure | HTTP error or timeout | Return citations + empty or fallback answer; set `error` string |

## 10. References

| Topic | Location | Anchor |
|-------|----------|--------|
| Strategic scope | [Gil Transcript Retrieval Blueprint](./gil-transcript-blueprint.md#2-first-milestone) | `2-first-milestone` |
| Retrieval-first rationale | [ADR 001](./adr/001-retrieval-first.md#decision) | `decision` |
| Evaluation plan (smoke queries, model/chunking checkpoints) | [Gil Transcript Evaluation](./gil-transcript-evaluation.md) | — |
| Phase-two follow-ons (rerank, clustering, RAG) | [Phase Two Roadmap](./phase-two-roadmap.md) | — |
| Scraper output format origin | [Scraper parser](../scripts/scrape_podscripts.py) | `scripts/scrape_podscripts.py` |
| Transcript directory | [Corpus path](../gil/transcripts) | `gil/transcripts` |
| Web chatbot ADR | [ADR 002](./adr/002-citation-first-web-chatbot.md) | — |
| Web app entry | [webapp/main.py](../webapp/main.py) | — |
