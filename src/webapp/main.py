# ruff: noqa: E402
"""Citation-first Gil transcript chatbot (FastAPI + Jinja2)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from neural.answer_markup import markdown_to_safe_html, plain_text_to_safe_html
from neural.chat_prompt import build_rag_messages
from neural.ingestion_status import summarize_ingestion_status
from neural.metadata_index import MetadataIndex, RetrievalFilters, load_metadata_index
from neural.openrouter import OpenRouterError, complete_chat, stream_chat
from neural.reranking import DEFAULT_RERANKER_MODEL, RerankerConfig
from neural.retrieval import (
    RetrievalBundle,
    citations_from_results,
    load_retrieval_bundle,
    retrieve,
)
from neural.vector_index import SearchResult

load_dotenv(_REPO_ROOT / ".env")

DEFAULT_INDEX_DIR = Path("data/transcript_index")
DEFAULT_METADATA_DIR = Path("data/metadata")
DEFAULT_FAILURE_LOG = Path("data/scrape_failures.log")
DEFAULT_MANIFEST_PATH = Path("gil/transcripts/manifest.json")
DEFAULT_TRANSCRIPTS_DIR = Path("gil/transcripts")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
logger = logging.getLogger(__name__)

MISSING_OPENROUTER_KEY_ANSWER = (
    "Set OPENROUTER_API_KEY to enable answer generation. Evidence excerpts are listed below."
)
MISSING_OPENROUTER_KEY_ERROR = "OPENROUTER_API_KEY not set"


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    index_dir = Path(os.environ.get("GIL_INDEX_DIR", str(DEFAULT_INDEX_DIR))).resolve()
    metadata_dir_raw = os.environ.get("GIL_METADATA_DIR", "").strip()
    metadata_dir = Path(metadata_dir_raw).resolve() if metadata_dir_raw else None
    try:
        app.state.bundle = load_retrieval_bundle(index_dir)
    except FileNotFoundError:
        app.state.bundle = None
    if metadata_dir is None and DEFAULT_METADATA_DIR.exists():
        metadata_dir = DEFAULT_METADATA_DIR.resolve()
    try:
        app.state.metadata_index = load_metadata_index(metadata_dir) if metadata_dir else None
    except FileNotFoundError:
        app.state.metadata_index = None
    app.state.index_dir = index_dir
    app.state.metadata_dir = metadata_dir
    app.state.started_at = time.time()
    app.state.metrics = {
        "request_count": 0,
        "error_count": 0,
        "chat_requests": 0,
        "stream_requests": 0,
        "last_request_ms": 0.0,
        "last_chat_ms": 0.0,
        "last_stream_ms": 0.0,
    }
    yield


app = FastAPI(title="Gil Transcript Chatbot", lifespan=lifespan)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_only: bool = False
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    episode_type: str | None = None
    guest_name: str | None = None
    speaker: str | None = None
    team: str | None = None
    topic: str | None = None
    source_file: str | None = None
    rerank: bool = False
    rerank_top_n: int = Field(default=20, ge=1, le=100)


class ChatResponse(BaseModel):
    answer: str
    answer_html: str | None = None
    citations: list[dict[str, Any]]
    generation_skipped: bool
    error: str | None = None


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    request.app.state.metrics["request_count"] += 1
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        request.app.state.metrics["error_count"] += 1
        logger.exception("request_failed request_id=%s path=%s", request_id, request.url.path)
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    request.app.state.metrics["last_request_ms"] = round(duration_ms, 3)
    if response.status_code >= 400:
        request.app.state.metrics["error_count"] += 1
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration_ms:.3f}ms"
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.3f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 3),
        },
    )
    return response


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


def _get_metadata_index(request: Request) -> MetadataIndex | None:
    metadata_index = getattr(request.app.state, "metadata_index", None)
    if isinstance(metadata_index, MetadataIndex):
        return metadata_index
    return None


def _index_loaded(request: Request) -> bool:
    return getattr(request.app.state, "bundle", None) is not None


def _metadata_loaded(request: Request) -> bool:
    return getattr(request.app.state, "metadata_index", None) is not None


def _retrieve_for_chat(
    request: Request,
    body: ChatRequest,
) -> tuple[list[SearchResult], list[dict[str, Any]]]:
    """Run retrieval and build citation payloads (shared by JSON and SSE chat)."""
    bundle = _get_bundle(request)
    metadata_index = _get_metadata_index(request)
    results = retrieve(
        bundle,
        body.message,
        top_k=body.top_k,
        model_override=None,
        metadata_index=metadata_index,
        filters=_build_filters(body),
        reranker=_build_reranker(body),
    )
    citations = citations_from_results(results, metadata_index)
    return results, citations


def _rag_messages(body: ChatRequest, results: list[SearchResult]) -> list[dict[str, str]]:
    history_payload = [t.model_dump() for t in body.history]
    return build_rag_messages(body.message, results, history=history_payload)


def _openrouter_api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return key or None


def _build_filters(body: ChatRequest) -> RetrievalFilters:
    return RetrievalFilters(
        episode_type=body.episode_type,
        guest_name=body.guest_name,
        speaker=body.speaker,
        team=body.team,
        topic=body.topic,
        source_file=body.source_file,
    )


def _build_reranker(body: ChatRequest) -> RerankerConfig | None:
    if not body.rerank:
        return None
    return RerankerConfig(
        model_name=os.environ.get("GIL_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        top_n=body.rerank_top_n,
    )


def _retrieval_only_text(citations: list[dict[str, Any]]) -> str:
    lines = [
        (
            f"{c['rank']}. {c['episode_title']} "
            f"[{c['start_timestamp']}-{c['end_timestamp']}] score={c['score']}"
        )
        for c in citations
    ]
    return "\n".join(lines) if lines else "No matching transcript excerpts found."


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "index_ok": _index_loaded(request),
            "metadata_ok": _metadata_loaded(request),
        },
    )


@app.get("/api/metrics")
async def api_metrics(request: Request) -> dict[str, Any]:
    """Compact JSON metrics for operators (distinct from /health)."""
    uptime_seconds = round(time.time() - request.app.state.started_at, 3)
    return {
        "started_at": request.app.state.started_at,
        "uptime_seconds": uptime_seconds,
        "metrics": dict(request.app.state.metrics),
    }


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    uptime_seconds = round(time.time() - request.app.state.started_at, 3)
    return {
        "status": "ok",
        "index_loaded": _index_loaded(request),
        "metadata_loaded": _metadata_loaded(request),
        "index_dir": str(getattr(request.app.state, "index_dir", DEFAULT_INDEX_DIR)),
        "metadata_dir": str(getattr(request.app.state, "metadata_dir", ""))
        if getattr(request.app.state, "metadata_dir", None)
        else None,
        "uptime_seconds": uptime_seconds,
        "metrics": dict(request.app.state.metrics),
    }


@app.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    if not _index_loaded(request):
        raise HTTPException(status_code=503, detail="Index not loaded")
    return {"status": "ready"}


@app.get("/api/ingestion/status")
async def ingestion_status(request: Request) -> dict[str, Any]:
    index_dir = getattr(request.app.state, "index_dir", DEFAULT_INDEX_DIR)
    metadata_dir = getattr(request.app.state, "metadata_dir", DEFAULT_METADATA_DIR)
    summary = summarize_ingestion_status(
        transcripts_dir=DEFAULT_TRANSCRIPTS_DIR,
        manifest_path=DEFAULT_MANIFEST_PATH,
        failure_log_path=DEFAULT_FAILURE_LOG,
        index_dir=index_dir,
        metadata_dir=metadata_dir if isinstance(metadata_dir, Path) else DEFAULT_METADATA_DIR,
    )
    return asdict(summary)


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: Request, body: ChatRequest) -> ChatResponse:
    request.app.state.metrics["chat_requests"] += 1
    t0 = time.perf_counter()
    try:
        results, citations = _retrieve_for_chat(request, body)

        if body.retrieval_only:
            answer = _retrieval_only_text(citations)
            return ChatResponse(
                answer=answer,
                answer_html=plain_text_to_safe_html(answer) or None,
                citations=citations,
                generation_skipped=True,
                error=None,
            )

        api_key = _openrouter_api_key()
        if api_key is None:
            return ChatResponse(
                answer=MISSING_OPENROUTER_KEY_ANSWER,
                answer_html=plain_text_to_safe_html(MISSING_OPENROUTER_KEY_ANSWER) or None,
                citations=citations,
                generation_skipped=True,
                error=MISSING_OPENROUTER_KEY_ERROR,
            )

        messages = _rag_messages(body, results)

        try:
            answer = complete_chat(
                messages,
                api_key=api_key,
                model=os.environ.get("OPENROUTER_MODEL"),
            )
        except OpenRouterError as exc:
            return ChatResponse(
                answer="",
                answer_html=None,
                citations=citations,
                generation_skipped=False,
                error=str(exc),
            )

        html = markdown_to_safe_html(answer) or None
        return ChatResponse(
            answer=answer,
            answer_html=html,
            citations=citations,
            generation_skipped=False,
            error=None,
        )
    finally:
        request.app.state.metrics["last_chat_ms"] = round(
            (time.perf_counter() - t0) * 1000,
            3,
        )


@app.post("/api/chat/stream")
async def api_chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    request.app.state.metrics["stream_requests"] += 1
    results, citations = _retrieve_for_chat(request, body)

    def event_stream():
        t0 = time.perf_counter()
        try:
            yield _sse_event("citations", {"citations": citations})
            if body.retrieval_only:
                answer = _retrieval_only_text(citations)
                yield _sse_event("answer", {"delta": answer})
                yield _sse_event(
                    "done",
                    {
                        "answer": answer,
                        "generation_skipped": True,
                        "answer_html": plain_text_to_safe_html(answer) or None,
                    },
                )
                return

            api_key = _openrouter_api_key()
            if api_key is None:
                yield _sse_event("answer", {"delta": MISSING_OPENROUTER_KEY_ANSWER})
                yield _sse_event(
                    "done",
                    {
                        "answer": MISSING_OPENROUTER_KEY_ANSWER,
                        "generation_skipped": True,
                        "error": MISSING_OPENROUTER_KEY_ERROR,
                        "answer_html": plain_text_to_safe_html(MISSING_OPENROUTER_KEY_ANSWER)
                        or None,
                    },
                )
                return

            messages = _rag_messages(body, results)
            answer_parts: list[str] = []
            try:
                for chunk in stream_chat(
                    messages,
                    api_key=api_key,
                    model=os.environ.get("OPENROUTER_MODEL"),
                ):
                    answer_parts.append(chunk)
                    yield _sse_event("answer", {"delta": chunk})
            except OpenRouterError as exc:
                partial = "".join(answer_parts)
                yield _sse_event("error", {"message": str(exc)})
                yield _sse_event(
                    "done",
                    {
                        "answer": partial,
                        "generation_skipped": False,
                        "error": str(exc),
                        "answer_html": markdown_to_safe_html(partial) if partial else None,
                    },
                )
                return

            answer = "".join(answer_parts)
            yield _sse_event(
                "done",
                {
                    "answer": answer,
                    "generation_skipped": False,
                    "answer_html": markdown_to_safe_html(answer) or None,
                },
            )
        finally:
            request.app.state.metrics["last_stream_ms"] = round(
                (time.perf_counter() - t0) * 1000,
                3,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
