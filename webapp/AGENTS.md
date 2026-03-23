# WEBAPP

## OVERVIEW
Minimal FastAPI + Jinja UI for citation-first chat over the local transcript index.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Startup/index loading | `webapp/main.py` | FastAPI lifespan loads bundle from `GIL_INDEX_DIR` |
| API schema | `webapp/main.py` | `ChatTurn`, `ChatRequest`, `ChatResponse` |
| Chat endpoint | `webapp/main.py` | `/api/chat` retrieval-only and generation branches |
| UI markup/JS | `webapp/templates/chat.html` | Single template, inline styles, inline fetch logic |

## CONVENTIONS
- The app loads `.env` from repo root at startup.
- Retrieval is local and mandatory; generation is optional and should degrade cleanly when `OPENROUTER_API_KEY` is missing.
- Evidence is first-class UI output, not debug detail.

## ANTI-PATTERNS
- Do not let the web app diverge from `neural.retrieval` behavior used by the CLI.
- Do not hide citations behind generation success; citations must still show in retrieval-only or error cases.
- Do not add persistent multi-user state or auth as a casual tweak; ADR 002 marks those as non-goals for v1.
- Do not store API keys in code or templates; environment only.

## NOTES
- `webapp/templates/chat.html` is intentionally simple and single-file; keep changes easy to audit.
- `tests/test_webapp.py` covers the main happy path, validation, missing key, and missing index cases.

## COMMANDS
- `uv run uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000`
- Build the index first if `GIL_INDEX_DIR` does not point at a valid bundle.
- Use retrieval-only mode to debug evidence output without involving OpenRouter.
