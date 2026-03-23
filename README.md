# neural

Utilities for indexing and querying **Gil's Arena** podcast transcripts. Place transcript `.txt` files under `gil/transcripts/` (or pass `--transcripts-dir`), then build a local FAISS index and run the citation-first RAG chatbot.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)

## Setup

```bash
uv sync
```

This installs runtime deps (`requests`, `beautifulsoup4`, `tqdm`, `sentence-transformers` for cross-encoder reranking, `faiss-cpu`, `numpy`, `fastapi`, `uvicorn`, `jinja2`) and the default **dev** group (`ruff`, `ty`, `pytest`). Use `uv sync --all-groups` if you add other dependency groups later.

## Usage

### Build A Semantic Index

Index builds and query-time dense retrieval use **OpenRouter embeddings** only. Set `OPENROUTER_API_KEY` and `OPENROUTER_EMBEDDING_MODEL` in `.env` (see [`.env.example`](.env.example)); [`scripts/build_transcript_index.py`](scripts/build_transcript_index.py) loads `.env` from the repo root automatically.

Build a local FAISS index over timestamped transcript chunks:

```bash
uv run python scripts/build_transcript_index.py
```

Useful flags:

```bash
# Build from a subset of episodes
uv run python scripts/build_transcript_index.py --limit 10

# Change chunk sizing
uv run python scripts/build_transcript_index.py --lines-per-chunk 6 --line-overlap 2

# Override the OpenRouter embedding model id (otherwise uses OPENROUTER_EMBEDDING_MODEL)
uv run python scripts/build_transcript_index.py --model mistralai/mistral-embed-2312 --output-dir data/transcript_index_alt

# Incremental update (same model + chunking as existing OpenRouter index)
uv run python scripts/build_transcript_index.py --incremental
```

Indexes built with older local sentence-transformers embeddings are no longer supported; run a **full** rebuild after upgrading.

### Query The Index

Queries embed the text via OpenRouter using the same API key and (by default) the `model_name` stored in `config.json`. Set env vars as for the build step; [`scripts/query_transcripts.py`](scripts/query_transcripts.py) loads `.env` from the repo root.

After building an index, search it with a free-text basketball query:

```bash
uv run python scripts/query_transcripts.py "What did they say about Giannis trade rumors?"
```

Useful flags:

```bash
uv run python scripts/query_transcripts.py "Team USA basketball" --top-k 3
uv run python scripts/query_transcripts.py "Bronny James" --index-dir data/transcript_index
uv run python scripts/query_transcripts.py "Lakers chemistry" --metadata-dir data/metadata --team Lakers
uv run python scripts/query_transcripts.py "Dwight Howard" --metadata-dir data/metadata --guest "Dwight Howard" --rerank
```

### Evaluate Retrieval

Run the tracked seed queries and compute hit rate and MRR:

```bash
uv run python scripts/evaluate_retrieval.py
uv run python scripts/evaluate_retrieval.py --metadata-dir data/metadata --rerank --json
uv run python scripts/evaluate_retrieval.py --hybrid --json
```

### Topic clustering (offline)

Cluster chunk embeddings and write `topics_report.json`:

```bash
uv run python scripts/cluster_transcript_chunks.py --index-dir data/transcript_index
```

### Artifact Status Report

Summarize transcript files on disk and index/metadata build state:

```bash
uv run python scripts/report_ingestion_status.py
uv run python scripts/report_ingestion_status.py --json
```

### Chat Web App (citation-first RAG)

After building the index, run the local FastAPI UI:

```bash
# Optional: copy env template and set secrets (loaded automatically from repo root)
cp .env.example .env   # then edit .env — .env is gitignored

uv run uvicorn --app-dir src webapp.main:app --reload --host 127.0.0.1 --port 8000
```

You can instead `export` the same variables listed in [`.env.example`](.env.example); see that file for `OPENROUTER_*` and `GIL_INDEX_DIR`.

Open `http://127.0.0.1:8000`. Toggle **Retrieval only** to skip the LLM.

The web UI also supports optional metadata-aware filters, cross-encoder reranking, optional hybrid (BM25 + dense) retrieval, streaming responses, `/health`, `/ready`, and `/api/ingestion/status`.

## Outputs

- **Transcripts**: one `.txt` per episode under `gil/transcripts` (or your `--transcripts-dir`).
- **Index artifacts**: local FAISS bundle under `data/transcript_index` by default:
  - `index.faiss`
  - `chunks.json`
  - `config.json`
  - `source_manifest.json` (when using incremental builds)

## Evaluation Seeds

- Evaluation seed queries: `evals/gil_queries.json`

## Contributing

```bash
uv run ruff check src scripts tests
uv run ty check
uv run pytest
```

Tests import from `src/`, the repo root, and `scripts/` with `pythonpath = ["src", ".", "scripts"]` in `pyproject.toml`.
