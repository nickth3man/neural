# Phase Two Roadmap: After Retrieval MVP

This document tracks **follow-on** work once semantic indexing and query CLI are validated. Nothing here is required for the first milestone.

## 1. Prerequisites

- Stable index build and query flow (see [blueprint](./gil-transcript-blueprint.md)).
- Smoke evaluation passing on [`evals/gil_queries.json`](../evals/gil_queries.json) (see [evaluation plan](./gil-transcript-evaluation.md)).

## 2. Reranking Path

**Goal:** Improve top-result ordering when embedding retrieval recall is acceptable but ranking is noisy.

**Approach:**

1. Retrieve top 20–50 chunks with the existing bi-encoder (sentence-transformers).
2. Rerank candidates with a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L6-v2`).
3. Expose `--rerank` or a separate script so baseline latency stays unchanged.

**References:** Official sentence-transformers retrieve-and-rerank examples in the upstream repo.

## 3. Clustering And Topic Discovery

**Goal:** Discover recurring themes across episodes without labels.

**Options:**

- **K-means / agglomerative** on chunk embeddings (sentence-transformers clustering examples).
- **BERTopic** with a `SentenceTransformer` backend and optional precomputed embeddings for reproducibility.

**Output:** Topic tables, episode-topic summaries, or static HTML reports (deferred).

## 4. Supervised Classification

**Goal:** Predict labels such as team, segment type (ad vs discussion), or “hot take” intensity.

**Requirements:**

- Manual labeling of hundreds of chunks (spreadsheet or simple JSON).
- PyTorch `Dataset` + `DataLoader` + small classifier (e.g. linear head on frozen embeddings, or fine-tuned encoder).

**Windows note:** Use `if __name__ == "__main__":` guards if `num_workers > 0` in `DataLoader`.

## 5. RAG Layer

**Goal:** Natural-language answers with citations to transcript chunks.

**Approach:**

1. Retrieve top-k chunks (existing index).
2. Pass retrieved text + user question to an LLM API or local model with strict “answer only from context” prompting.
3. Always surface `source_file` and timestamp range in the UI or stdout.

**Risk:** Hallucination; mitigate with citation-only answers and low temperature.

### 5.1 Web App MVP (implemented)

- **Entry:** `uv run uvicorn webapp.main:app --reload` (from repo root; see [README](../README.md)).
- **Index:** Read-only load of `data/transcript_index` (or `GIL_INDEX_DIR`); same artifacts as the CLI.
- **Generation:** OpenRouter (`OPENROUTER_API_KEY`, optional `OPENROUTER_MODEL`); retrieval uses local `sentence-transformers` + FAISS only.
- **UX:** Answer panel plus an evidence panel listing each hit with rank, score, episode, file, timestamps, and chunk text. **Retrieval-only** toggle skips the LLM.
- **Security:** API key via environment only; no multi-tenant storage in v1.

### 5.2 Minimal RAG evaluation

See [Gil Transcript Evaluation](./gil-transcript-evaluation.md#8-rag--web-ui-smoke): manual checks that answers cite shown chunks and refuse when context is insufficient.

## 6. Generative / Style Modeling

**Goal:** Gil’s Arena–style continuation or summarization.

**Status:** Deferred. Corpus size favors **fine-tuning** or **prompting** over training a language model from scratch.

## 7. References

| Topic | Location |
|-------|----------|
| ADR (retrieval-first) | [adr/001-retrieval-first.md](./adr/001-retrieval-first.md) |
| Tech spec | [gil-transcript-tech-spec.md](./gil-transcript-tech-spec.md) |
| Evaluation | [gil-transcript-evaluation.md](./gil-transcript-evaluation.md) |
| Web chatbot ADR | [adr/002-citation-first-web-chatbot.md](./adr/002-citation-first-web-chatbot.md) |
