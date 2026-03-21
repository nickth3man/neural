"""Citation-first Gil transcript chatbot (FastAPI + Jinja2)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from neural.chat_prompt import build_rag_messages
from neural.openrouter import OpenRouterError, complete_chat
from neural.retrieval import (
    RetrievalBundle,
    citations_from_results,
    load_retrieval_bundle,
    retrieve,
)
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

DEFAULT_INDEX_DIR = Path("data/transcript_index")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_dir = Path(os.environ.get("GIL_INDEX_DIR", str(DEFAULT_INDEX_DIR))).resolve()
    try:
        app.state.bundle = load_retrieval_bundle(index_dir)
    except FileNotFoundError:
        app.state.bundle = None
    app.state.index_dir = index_dir
    yield


app = FastAPI(title="Gil Transcript Chatbot", lifespan=lifespan)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_only: bool = False
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    generation_skipped: bool
    error: str | None = None


def _get_bundle(request: Request) -> RetrievalBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        idx = getattr(request.app.state, "index_dir", DEFAULT_INDEX_DIR)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Transcript index not found at {idx}. "
                "Run: uv run python scripts/build_transcript_index.py"
            ),
        )
    return bundle


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"index_ok": getattr(request.app.state, "bundle", None) is not None},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: Request, body: ChatRequest) -> ChatResponse:
    bundle = _get_bundle(request)
    results = retrieve(
        bundle,
        body.message,
        top_k=body.top_k,
        model_override=None,
    )
    citations = citations_from_results(results)

    if body.retrieval_only:
        lines = [
            f"{c['rank']}. {c['episode_title']} [{c['start_timestamp']}-{c['end_timestamp']}] "
            f"score={c['score']}"
            for c in citations
        ]
        text = "\n".join(lines) if lines else "No matching transcript excerpts found."
        return ChatResponse(
            answer=text,
            citations=citations,
            generation_skipped=True,
            error=None,
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return ChatResponse(
            answer=(
                "Set OPENROUTER_API_KEY to enable answer generation. "
                "Evidence excerpts are listed below."
            ),
            citations=citations,
            generation_skipped=True,
            error="OPENROUTER_API_KEY not set",
        )

    history_payload = [t.model_dump() for t in body.history]
    messages = build_rag_messages(body.message, results, history=history_payload)

    try:
        answer = complete_chat(
            messages,
            api_key=api_key,
            model=os.environ.get("OPENROUTER_MODEL"),
        )
    except OpenRouterError as exc:
        return ChatResponse(
            answer="",
            citations=citations,
            generation_skipped=False,
            error=str(exc),
        )

    return ChatResponse(
        answer=answer,
        citations=citations,
        generation_skipped=False,
        error=None,
    )
