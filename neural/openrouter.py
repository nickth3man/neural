"""Minimal OpenRouter chat-completions client (OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class OpenRouterError(Exception):
    """Raised when OpenRouter returns an error or unexpected payload."""


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
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://github.com/local/neural"),
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
