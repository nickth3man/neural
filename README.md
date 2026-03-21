# neural

Utilities for scraping, indexing, and querying **Gil's Arena** podcast transcripts from [PodScripts.co](https://podscripts.co).

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)

## Setup

```bash
uv sync
```

This installs runtime deps (`requests`, `beautifulsoup4`, `tqdm`, `sentence-transformers`, `faiss-cpu`, `numpy`, `fastapi`, `uvicorn`, `jinja2`) and the default **dev** group (`ruff`, `ty`, `pytest`). Use `uv sync --all-groups` if you add other dependency groups later.

## Usage

### Scrape Transcripts

Run the scraper with Python (from the repo root):

```bash
# Quick test: few episodes, single listing page
uv run python scripts/scrape_podscripts.py --dry-run

# Full scrape (default output: gil/transcripts)
uv run python scripts/scrape_podscripts.py

# Skip episodes that already have a matching .txt in the output dir
uv run python scripts/scrape_podscripts.py --resume

# Page range and cap
uv run python scripts/scrape_podscripts.py --start-page 1 --end-page 5 --limit 10

# Debug logging
uv run python scripts/scrape_podscripts.py --verbose
```

For every option, see:

```bash
uv run python scripts/scrape_podscripts.py --help
```

### Build A Semantic Index

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

# Override the embedding model or output directory
uv run python scripts/build_transcript_index.py --model multi-qa-mpnet-base-cos-v1 --output-dir data/transcript_index_mpnet
```

### Query The Index

After building an index, search it with a free-text basketball query:

```bash
uv run python scripts/query_transcripts.py "What did they say about Giannis trade rumors?"
```

Useful flags:

```bash
uv run python scripts/query_transcripts.py "Team USA basketball" --top-k 3
uv run python scripts/query_transcripts.py "Bronny James" --index-dir data/transcript_index
```

### Chat Web App (citation-first RAG)

After building the index, run the local FastAPI UI:

```bash
# Optional: copy env template and set secrets (loaded automatically from repo root)
cp .env.example .env   # then edit .env — .env is gitignored

uv run uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
```

You can instead `export` the same variables listed in [`.env.example`](.env.example); see that file for `OPENROUTER_*` and `GIL_INDEX_DIR`.

Open `http://127.0.0.1:8000`. Toggle **Retrieval only** to skip the LLM. See [ADR 002](docs/adr/002-citation-first-web-chatbot.md) and [Phase Two Roadmap](docs/phase-two-roadmap.md).

### Manifest (this run only)

Write a JSON array of successfully scraped episodes (metadata + transcript URL + path relative to `--output-dir`):

```bash
uv run python scripts/scrape_podscripts.py --manifest gil/transcripts/manifest.json
```

If there are no episodes to scrape or none succeed, the file is still written as `[]`. The manifest is **not** merged with previous runs; combine files yourself if you need a full index.

## Outputs

- **Transcripts**: one `.txt` per episode under `--output-dir` (default `gil/transcripts`).
- **Failures**: append-only log at `data/scrape_failures.log` (default; see `ScraperConfig` in `scripts/scrape_podscripts.py`).
- **Manifest**: optional JSON when `--manifest` is set.
- **Index artifacts**: local FAISS bundle under `data/transcript_index` by default:
  - `index.faiss`
  - `chunks.json`
  - `config.json`

## Retrieval Docs

- Strategic blueprint: `docs/gil-transcript-blueprint.md`
- Technical spec: `docs/gil-transcript-tech-spec.md`
- ADR: `docs/adr/001-retrieval-first.md`
- Evaluation plan: `docs/gil-transcript-evaluation.md`
- Phase-two roadmap: `docs/phase-two-roadmap.md`
- Evaluation seed queries: `evals/gil_queries.json`

## Etiquette

The script checks `robots.txt`, waits between list-page and episode requests (default **1s**, override with `--delay SEC`), and retries with backoff. Tune `rate_limit_seconds`, `total_pages`, and related fields in `ScraperConfig` inside `scripts/scrape_podscripts.py` if the site policy or layout changes.

## Contributing

```bash
uv run ruff check neural scripts tests webapp
uv run ty check
uv run pytest
```

Tests import both the scraper module and the `neural` package with `pythonpath = [".", "scripts"]` in `pyproject.toml`.
