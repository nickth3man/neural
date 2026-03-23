# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-21
**Commit:** 8cc09a3
**Branch:** master

## OVERVIEW
Python 3.12 retrieval stack for Gil's Arena transcripts: scrape PodScripts pages, build a local FAISS index, query it from CLI, and optionally serve a citation-first FastAPI chat UI.

## STRUCTURE
```text
./
|- neural/        core transcript, embedding, retrieval, and OpenRouter modules
|- scripts/       executable pipeline entry points
|- webapp/        FastAPI app plus Jinja template
|- tests/         flat pytest suite
|- docs/          blueprint, tech spec, ADRs, roadmap
|- evals/         seed evaluation queries
|- gil/           scraped transcript corpus; treat as content, not source code
`- data/          generated artifacts and logs
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Parse transcript text | `neural/corpus.py` | Timestamp parsing, continuation-line merge rules |
| Define chunk windows | `neural/chunking.py` | Deterministic overlapping windows |
| Build or load index artifacts | `neural/vector_index.py` | `index.faiss`, `chunks.json`, `config.json` |
| Run retrieval in code | `neural/retrieval.py` | Shared path used by CLI and web app |
| Call OpenRouter | `neural/openrouter.py` | Stdlib `urllib`, no SDK |
| Scrape transcripts | `scripts/scrape_podscripts.py` | Largest script; robots, retries, manifest |
| Build index from corpus | `scripts/build_transcript_index.py` | `gil/transcripts` -> `data/transcript_index` |
| Extract metadata | `scripts/extract_metadata.py` | Deterministic + optional LLM enrichment |
| Serve chat UI | `webapp/main.py` | Loads `.env`, index bundle, `/api/chat` |
| Validate behavior | `tests/` | Flat `test_*.py`, heavy monkeypatch/tmp_path usage |
| Understand design intent | `docs/adr/` | Retrieval-first + citation-first decisions |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `ScraperConfig` | dataclass | `scripts/scrape_podscripts.py` | Scraper defaults, paths, retry/rate-limit knobs |
| `run_scraper` | function | `scripts/scrape_podscripts.py` | Main scrape flow |
| `ChunkingConfig` | dataclass | `neural/chunking.py` | Line-window chunk policy |
| `TranscriptDocument` | dataclass | `neural/corpus.py` | Structured transcript file |
| `SearchResult` | dataclass | `neural/vector_index.py` | Ranked retrieval hit |
| `RetrievalBundle` | dataclass | `neural/retrieval.py` | In-memory index + chunk metadata + config |
| `complete_chat` | function | `neural/openrouter.py` | Hosted LLM call |
| `app` | FastAPI app | `webapp/main.py` | Web entry point |

## CONVENTIONS
- Use `uv`, not `pip`/Poetry; CI installs with `uv sync --all-groups`.
- Keep code Python 3.12-friendly; repo pins 3.12 in `.python-version` and `pyproject.toml`.
- Ruff is intentionally narrow: `E,F,I,UP`; line length is 100.
- Type checking goes through `ty`; `mypy.ini` mainly exists for editors.
- Tests may import from both repo root and `scripts/`; `pythonpath = [".", "scripts"]` is deliberate.
- `webapp/main.py` loads `.env` from repo root; secrets belong in `.env`, never in tracked files.

## ANTI-PATTERNS (THIS PROJECT)
- Do not treat whole episodes as the first retrieval unit; chunk by timestamped windows.
- Do not switch cosine-style retrieval to unnormalized embeddings or `IndexFlatL2`; docs standardize on normalized embeddings plus `IndexFlatIP`.
- Do not fail corpus loading on a single malformed line; parser rules prefer skipping or merging untimed text.
- Do not download models in unit tests; tests use synthetic embeddings and `pytest.importorskip("faiss")`.
- Do not invent speaker names or open-ended labels in metadata enrichment; several modules rely on closed vocabularies and roster-constrained attribution.
- Do not bypass citations in chat flows; the web app is citation-first and supports retrieval-only mode by design.

## UNIQUE STYLES
- Scripts are first-class entry points, not disposable one-offs; tests import them directly.
- The repo separates source corpus (`gil/transcripts`) from derived outputs (`data/`).
- ADRs are current and should shape implementation choices before new architecture work.

## COMMANDS
```bash
uv sync
uv sync --all-groups
uv run ruff check neural scripts tests webapp
uv run ty check
uv run pytest
uv run python scripts/scrape_podscripts.py --help
uv run python scripts/build_transcript_index.py
uv run python scripts/query_transcripts.py "Team USA basketball"
uv run uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
```

## NOTES
- Ignore `.venv/`, caches, `data/`, and `gil/transcripts/` when reasoning about code structure.
- `scripts/scrape_podscripts.py` is the main complexity hotspot; start there for scraping changes.
- `evals/gil_queries.json` is reference data, not a code boundary; root coverage is enough.
