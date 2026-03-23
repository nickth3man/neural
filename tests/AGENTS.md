# TESTS

## OVERVIEW
Flat pytest suite covering library modules, CLI helpers, and the FastAPI surface.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Chunking expectations | `tests/test_chunking.py` | Deterministic overlap/timestamp assertions |
| Retrieval bundle behavior | `tests/test_retrieval.py` | Synthetic FAISS bundle + monkeypatching |
| Web app behavior | `tests/test_webapp.py` | TestClient, env mutation, temp index |
| Hosted chat client | `tests/test_openrouter.py` | Mocked `urllib` responses |
| Scraper parsing | `tests/test_parse_episode_list.py`, `tests/test_parse_transcript_page.py` | HTML parsing expectations |

## CONVENTIONS
- Keep tests function-first; shared fixtures are local to each file rather than centralized in `conftest.py`.
- Use `tmp_path` for index artifacts and other filesystem outputs.
- Use monkeypatching or mocks for embeddings, HTTP calls, and environment variables.
- Keep return annotations on test functions (`-> None`) to match codebase style.

## ANTI-PATTERNS
- Do not download models or rely on network access in tests.
- Do not require FAISS unconditionally; tests that need it already use `pytest.importorskip("faiss")`.
- Do not assume scripts are packaged modules; tests intentionally import from `scripts/` because pytest adds that path.

## NOTES
- There is no `conftest.py`; if a fixture is only useful once, keep it local.
- File names mirror the behavior under test closely enough that root-level navigation stays fast.

## COMMANDS
- `uv run pytest`
- `uv run pytest tests/test_retrieval.py`
