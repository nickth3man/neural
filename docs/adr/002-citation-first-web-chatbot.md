# ADR 002: Citation-First Web Chatbot (Phase Two)

## Status

Accepted

## Context

ADR 001 established a retrieval-first CLI MVP. The corpus is indexed locally with FAISS; queries return timestamped transcript chunks. Users now want a conversational surface that summarizes answers while preserving evidence.

## Decision

Add an optional **local web chatbot** that:

1. Uses the **same retrieval path** as `scripts/query_transcripts.py` (load bundle → embed query → FAISS search → ranked `SearchResult` list).
2. Sends retrieved chunks + user message to a **hosted LLM via OpenRouter** for generation.
3. **Always** shows retrieval hits (file, episode title, timestamps, chunk text) as first-class citations in the UI.
4. Supports **retrieval-only mode** (no LLM call) when the user toggles it or when the API key is missing / the call fails.

The web app does **not** replace the CLI for validation; smoke checks on `evals/gil_queries.json` remain the retrieval bar.

## Consequences

### Positive

- Natural-language answers with visible provenance.
- Retrieval remains the system of truth; the LLM is a thin synthesis layer.
- Local index path; no transcript exfiltration beyond what the operator already has on disk.

### Negative

- Requires `OPENROUTER_API_KEY` and network for full chat mode.
- Hallucination risk if prompts drift; mitigated by strict system instructions and citation UI.

### Non-goals (v1)

- Multi-user auth, persistent chat history in a database, reranking, or remote embedding APIs.

## References

| Topic | Location |
|-------|----------|
| Retrieval-first baseline | [ADR 001](./001-retrieval-first.md) |
| RAG roadmap detail | [Phase Two Roadmap](../phase-two-roadmap.md) |
| Evaluation (RAG smoke) | [Gil Transcript Evaluation](../gil-transcript-evaluation.md) |
