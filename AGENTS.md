# AGENTS.md

This file provides guidance for AI agents operating in this repository.

## Project Overview

**neural** is a Python 3.12+ project for indexing and querying Gil's Arena podcast transcripts (static `.txt` corpus). It provides a citation-first RAG chatbot via FastAPI.

## Build & Development Commands

### Setup

```bash
uv sync              # Install all dependencies (runtime + dev)
uv sync --all-groups # Install all dependency groups
```

### Linting & Type Checking

```bash
uv run ruff check src scripts tests    # Lint with ruff (E, F, I, UP rules)
uv run ty check                        # Strict type checking with ty
uv run mypy src scripts tests          # mypy type checking
```

### Testing

```bash
uv run pytest                           # Run all tests
uv run pytest tests/test_chunking.py    # Run single test file
uv run pytest tests/ -k "chunking"      # Run tests matching pattern
uv run pytest tests/ --lf              # Run only last failed tests
```

### Running the Web App

```bash
uv run uvicorn --app-dir src webapp.main:app --reload --host 127.0.0.1 --port 8000
```

### Chat UI (TypeScript)

The browser client is authored in TypeScript and compiled to static JavaScript.

```bash
cd src/webapp/frontend && npm install && npm run build
```

This emits [`src/webapp/static/chat.js`](src/webapp/static/chat.js). Commit the compiled file so the app runs without Node; rebuild after editing [`chat.ts`](src/webapp/frontend/chat.ts).

### Pre-commit Hooks

Hooks run automatically on commit (ruff check, ty check, pytest). Install with:

```bash
pre-commit install
```

## Code Style Guidelines

### General

- **Python version**: 3.12+ (required)
- **Line length**: 100 characters
- **Indent**: 4 spaces (or as configured in editors)
- **String quotes**: Double quotes preferred; single quotes when avoiding escape

### Imports

- Always use `from __future__ import annotations` for forward references
- Order imports: stdlib → third-party → local (enforced by ruff I rule)
- Use absolute imports from package root: `from neural.chunking import ...`
- Avoid wildcard imports (`from x import *`)

### Type Annotations

- Use modern union syntax: `X | None` not `Optional[X]`
- Use `object` for JSON-serializable dict values
- Prefer explicit type hints on function signatures
- Return types: Always annotate public functions
- Generic types: `list[X]`, `dict[K, V]`, `tuple[X, ...]`

### Dataclasses

- Use `frozen=True` for immutable data structures
- Use `slots=True` for memory efficiency
- Validate in `__post_init__` with clear error messages

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class MyData:
    name: str
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            msg = "value must be non-negative"
            raise ValueError(msg)
```

### Enums

- Inherit from `str, Enum` for JSON-serializable enums
- Use SCREAMING_SNAKE_CASE for members
- Always provide string values matching the member name

```python
class EpisodeType(str, Enum):
    SEASON_PREMIERE = "season_premiere"
    GUEST_INTERVIEW = "guest_interview"
```

### Error Handling

- Use `msg = "..."` then `raise XError(msg)` pattern for clarity
- Never suppress errors silently (`except: pass`)
- Prefer specific exceptions (ValueError, FileNotFoundError)
- Custom exceptions should inherit from Exception

```python
if top_k < 1:
    msg = "top_k must be at least 1"
    raise ValueError(msg)
```

### Naming Conventions

| Thing | Convention | Example |
| ----- | ---------- | ------- |
| Functions/variables | snake_case | `chunk_transcript`, `episode_title` |
| Classes/types | PascalCase | `TranscriptChunk`, `RetrievalBundle` |
| Constants | SCREAMING_SNAKE_CASE | `DEFAULT_INDEX_DIR` |
| Private members | _leading_underscore | `_internal_state` |
| Enum members | SCREAMING_SNAKE_CASE | `EpisodeType.SEASON_PREMIERE` |

### Docstrings

- Use triple double quotes `"""`
- Write docstrings for all public modules, classes, and functions
- Use Google-style or Napoleon-compatible docstrings

```python
def chunk_transcript(
    document: TranscriptDocument,
    config: ChunkingConfig = ChunkingConfig(),
) -> list[TranscriptChunk]:
    """Split a transcript document into overlapping retrieval chunks.

    Args:
        document: The transcript to chunk.
        config: Chunking configuration with line size and overlap.

    Returns:
        List of TranscriptChunk objects ordered by position.
    """
```

### FastAPI/Webapp Patterns

- Use Pydantic `BaseModel` for request/response schemas
- Use `Field()` for validation constraints
- Use `HTTPException` for HTTP errors
- Use async context managers for lifespan events
- Store app state in `request.app.state`

```python
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
```

## Project Structure

```text
neural/
├── src/
│   ├── neural/           # Core library code
│   │   ├── chunking.py   # Transcript chunking logic
│   │   ├── corpus.py     # Corpus loading
│   │   ├── retrieval.py  # Retrieval logic
│   │   └── ...
│   └── webapp/           # FastAPI web application
│       ├── main.py       # App entry point
│       └── templates/    # Jinja2 templates
├── scripts/              # Standalone CLI scripts
├── tests/                # pytest test suite
├── gil/transcripts/      # Local transcript .txt files (gitignored)
├── data/                 # Index and metadata (gitignored)
└── evals/                # Evaluation queries
```

## Linting Configuration

Ruff is configured in `pyproject.toml`:

- **target-version**: py312
- **line-length**: 100
- **select**: E, F, I, UP
- **src paths**: src, scripts, tests

Run `uv run ruff check src scripts tests --fix` to auto-fix issues.

## Testing Guidelines

- Tests live in `tests/` directory
- Use `pytest` with `pythonpath` configured in `pyproject.toml`
- Test files: `test_<module_name>.py`
- Test functions: `test_<description>()`
- Use type annotations on test functions
- Use `pytest.importorskip` for optional dependencies

## Environment & Secrets

- Copy `.env.example` to `.env` for local development
- `.env` is gitignored—never commit secrets
- Environment variables: `OPENROUTER_API_KEY`, `GIL_INDEX_DIR`, `GIL_METADATA_DIR`
