"""Build OpenAI-compatible chat messages for citation-first RAG."""

from __future__ import annotations

from typing import Any

from neural.vector_index import SearchResult

SYSTEM_PROMPT = """You are a helpful assistant for the Gil's Arena podcast transcript corpus.

Rules:
- Answer ONLY using information that appears in the provided transcript excerpts below.
- If the excerpts do not contain enough information, say so clearly and suggest what is missing.
- After your answer, list which excerpt numbers (1, 2, …) you relied on.
  If none apply, say "No excerpts support this answer."
- Do not invent quotes, dates, trades, or events not present in the excerpts.
- Keep a conversational tone but stay faithful to the source material.
"""


def format_context_block(results: list[SearchResult]) -> str:
    """Format ranked chunks as numbered context for the model."""
    if not results:
        return "(No transcript excerpts were retrieved for this query.)"

    parts: list[str] = []
    for result in results:
        chunk = result.chunk
        parts.append(
            f"[Excerpt {result.rank}] (score={result.score:.4f}) "
            f"{chunk.episode_title} | {chunk.source_file} | "
            f"{chunk.start_timestamp}-{chunk.end_timestamp}\n"
            f"{chunk.chunk_text}"
        )
    return "\n\n".join(parts)


def build_rag_messages(
    user_message: str,
    results: list[SearchResult],
    *,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """
    Build ``messages`` for a chat completions API.

    ``history`` entries must be ``{"role": "user"|"assistant", "content": str}``.
    Only the latest user turn is paired with fresh retrieval context.
    """
    context = format_context_block(results)
    user_with_context = (
        f"Transcript excerpts (use only these as evidence):\n\n{context}\n\n"
        f"User question:\n{user_message}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_with_context})
    return messages
