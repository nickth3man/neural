"""Tests for OpenRouter client (mocked HTTP)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from neural.openrouter import OpenRouterError, complete_chat, stream_chat


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._body.splitlines(keepends=True))


def test_complete_chat_success() -> None:
    payload = {"choices": [{"message": {"content": "  Hello world  "}}]}
    body = json.dumps(payload).encode()

    with patch("neural.openrouter.urllib.request.urlopen", return_value=_FakeResponse(body)):
        out = complete_chat([{"role": "user", "content": "hi"}], api_key="sk-test")

    assert out == "Hello world"


def test_complete_chat_http_error() -> None:
    import io
    import urllib.error
    from email.message import Message

    fp = io.BytesIO(b'{"error":"bad"}')
    err = urllib.error.HTTPError("https://openrouter.ai", 401, "Unauthorized", Message(), fp)

    with patch("neural.openrouter.urllib.request.urlopen", side_effect=err):
        with pytest.raises(OpenRouterError, match="OpenRouter HTTP 401"):
            complete_chat([{"role": "user", "content": "x"}], api_key="k")


def test_complete_chat_bad_shape() -> None:
    body = json.dumps({"choices": []}).encode()
    with patch("neural.openrouter.urllib.request.urlopen", return_value=_FakeResponse(body)):
        with pytest.raises(OpenRouterError, match="Unexpected OpenRouter"):
            complete_chat([{"role": "user", "content": "x"}], api_key="k")


def test_complete_chat_empty_key() -> None:
    with pytest.raises(OpenRouterError, match="empty"):
        complete_chat([], api_key="   ")


def test_stream_chat_yields_content_chunks() -> None:
    body = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    with patch("neural.openrouter.urllib.request.urlopen", return_value=_FakeResponse(body)):
        chunks = list(stream_chat([{"role": "user", "content": "hi"}], api_key="sk-test"))
    assert chunks == ["Hello", " world"]
