"""Tests for RAG prompt construction."""

from neural.chat_prompt import SYSTEM_PROMPT, build_rag_messages, format_context_block
from neural.chunking import TranscriptChunk
from neural.vector_index import SearchResult


def _chunk(text: str, title: str = "Ep", fname: str = "f.txt") -> TranscriptChunk:
    return TranscriptChunk(
        episode_title=title,
        source_file=fname,
        start_timestamp="00:01:00",
        end_timestamp="00:01:05",
        start_seconds=60,
        end_seconds=65,
        chunk_text=text,
        line_count=2,
    )


def test_format_context_block_empty() -> None:
    assert "No transcript excerpts" in format_context_block([])


def test_build_rag_messages_includes_context_and_question() -> None:
    results = [
        SearchResult(rank=1, score=0.9, chunk=_chunk("Giannis trade talk")),
    ]
    messages = build_rag_messages("What about Giannis?", results)
    assert messages[0]["role"] == "system"
    assert SYSTEM_PROMPT in messages[0]["content"]
    user = messages[-1]["content"]
    assert "Giannis trade talk" in user
    assert "What about Giannis?" in user
    assert "[Excerpt 1]" in user


def test_build_rag_messages_with_history() -> None:
    results = [SearchResult(rank=1, score=0.5, chunk=_chunk("only evidence"))]
    history = [{"role": "user", "content": "Earlier?"}, {"role": "assistant", "content": "Yes."}]
    messages = build_rag_messages("Follow up", results, history=history)
    assert len(messages) == 4
    assert messages[1]["role"] == "user" and messages[1]["content"] == "Earlier?"
    assert messages[2]["role"] == "assistant"
