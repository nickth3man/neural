# SCRIPTS

## OVERVIEW
Operational entry points for the transcript pipeline: scrape source content, build/query the retrieval index, and run metadata extraction.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Scraper behavior | `scripts/scrape_podscripts.py` | Robots, retries, manifests, output paths |
| Index build flow | `scripts/build_transcript_index.py` | Corpus -> chunks -> embeddings -> FAISS |
| CLI querying | `scripts/query_transcripts.py` | Thin wrapper over `neural.retrieval` |
| Metadata pipeline | `scripts/extract_metadata.py` | Optional OpenRouter enrichment |

## CONVENTIONS
- Scripts self-bootstrap imports from repo root; the `sys.path` adjustment plus `# ruff: noqa: E402` is intentional here.
- Keep argparse defaults aligned with README examples and actual artifact locations.
- Paths should stay configurable via CLI flags rather than hardcoded in library modules.

## ANTI-PATTERNS
- Do not remove robots/rate-limit safeguards from the scraper.
- Do not hardcode transcript or output directories across multiple scripts; flags and shared defaults should stay authoritative.
- Do not move retrieval logic into CLI print code; keep reusable behavior in `neural/`.
- Do not make LLM metadata extraction mandatory; `--skip-llm` and deterministic fallback are part of the workflow.

## NOTES
- `scripts/scrape_podscripts.py` is the largest file in the repo and the first place to inspect for scraping regressions.
- `scripts/build_transcript_index.py` is intentionally thin; major retrieval changes should usually land in `neural/` first.

## COMMANDS
- `uv run python scripts/scrape_podscripts.py --help`
- `uv run python scripts/build_transcript_index.py`
- `uv run python scripts/query_transcripts.py "Team USA basketball"`
