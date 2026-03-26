"""Minimal OpenRouter chat-completions client (OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class OpenRouterError(Exception):
    """Raised when OpenRouter returns an error or unexpected payload."""


def embed_texts_openrouter(
    texts: list[str],
    *,
    model: str,
    api_key: str,
    batch_size: int = 32,
    timeout_seconds: float = 300.0,
    max_retries: int = 5,
    initial_backoff: float = 5.0,
) -> list[list[float]]:
    """
    Call OpenRouter's OpenAI-compatible embeddings API for each batch of texts.

    Returns one embedding vector per input text (same order as ``texts``).
    Retries on timeout and URLError with exponential backoff.
    """
    if not api_key.strip():
        msg = "OpenRouter API key is empty"
        raise OpenRouterError(msg)
    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ValueError(msg)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get(
            "OPENROUTER_HTTP_REFERER", "https://github.com/local/neural"
        ),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Gil Transcript Chatbot"),
    }

    all_rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload: dict[str, Any] = {"model": model, "input": batch}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OPENROUTER_EMBEDDINGS_URL,
            data=body,
            method="POST",
            headers=headers,
        )

        raw: str | None = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < max_retries - 1:
                    backoff = initial_backoff * (2**attempt)
                    time.sleep(backoff)
                else:
                    msg = (
                        f"OpenRouter embeddings request failed after {max_retries} attempts: {exc}"
                    )
                    raise OpenRouterError(msg) from exc

        assert raw is not None, "raw must be assigned before the loop breaks"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = "OpenRouter embeddings returned non-JSON response"
            raise OpenRouterError(msg) from exc

        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(batch):
            msg = f"Unexpected OpenRouter embeddings payload: {raw[:400]}"
            raise OpenRouterError(msg)

        by_index: dict[int, list[float]] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            emb = item.get("embedding")
            if isinstance(idx, int) and isinstance(emb, list):
                by_index[idx] = [float(x) for x in emb]

        for i in range(len(batch)):
            vec = by_index.get(i)
            if vec is None:
                msg = "OpenRouter embeddings response missing an index"
                raise OpenRouterError(msg)
            all_rows.append(vec)

    return all_rows


def complete_chat(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str | None = None,
    timeout_seconds: float = 90.0,
    temperature: float = 0.3,
) -> str:
    """
    Call OpenRouter chat completions and return the assistant message content.

    Uses stdlib ``urllib`` to avoid extra runtime dependencies.
    """
    if not api_key.strip():
        msg = "OpenRouter API key is empty"
        raise OpenRouterError(msg)

    resolved_model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_HTTP_REFERER", "https://github.com/local/neural"
            ),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Gil Transcript Chatbot"),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        msg = f"OpenRouter HTTP {exc.code}: {detail[:500]}"
        raise OpenRouterError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"OpenRouter request failed: {exc}"
        raise OpenRouterError(msg) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = "OpenRouter returned non-JSON response"
        raise OpenRouterError(msg) from exc

    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        msg = f"Unexpected OpenRouter response shape: {raw[:300]}"
        raise OpenRouterError(msg) from exc

    if not isinstance(content, str):
        msg = "OpenRouter message content is not a string"
        raise OpenRouterError(msg)

    return content.strip()


def stream_chat(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str | None = None,
    timeout_seconds: float = 90.0,
    temperature: float = 0.3,
) -> Iterator[str]:
    if not api_key.strip():
        msg = "OpenRouter API key is empty"
        raise OpenRouterError(msg)

    resolved_model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_HTTP_REFERER", "https://github.com/local/neural"
            ),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Gil Transcript Chatbot"),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_line = line[6:]
                if payload_line == "[DONE]":
                    break
                try:
                    data = json.loads(payload_line)
                except json.JSONDecodeError:
                    continue
                try:
                    choice = data["choices"][0]
                except (KeyError, IndexError, TypeError) as exc:
                    msg = f"Unexpected OpenRouter stream shape: {payload_line[:300]}"
                    raise OpenRouterError(msg) from exc
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        msg = f"OpenRouter HTTP {exc.code}: {detail[:500]}"
        raise OpenRouterError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"OpenRouter request failed: {exc}"
        raise OpenRouterError(msg) from exc
