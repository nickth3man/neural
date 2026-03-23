# Gil Transcript Retrieval Evaluation Plan

This document defines how to validate retrieval quality after the MVP index is built. It complements the technical specification test matrix (`TC-*`, `IT-*`) with **human-in-the-loop** checks on real corpus data.

## 1. Seed Query Set

Tracked file: [`evals/gil_queries.json`](../evals/gil_queries.json).

Each entry contains:

| Field | Purpose |
|-------|---------|
| `query` | Natural-language prompt for `scripts/query_transcripts.py` |
| `expected_episode_substring` | Substring of the episode filename (stem) that should appear in top-k results |
| `notes` | Optional context for the evaluator |

## 2. Manual Smoke Procedure

Prerequisites: built index (`uv run python scripts/build_transcript_index.py`).

For each query in `gil_queries.json`:

1. Run: `uv run python scripts/query_transcripts.py "<query>" --top-k 5`
2. Pass if **at least one** of the top-5 `file=` lines contains `expected_episode_substring`.
3. Record failures in a scratch log (not committed) with query, expected substring, and actual top-5 filenames.

## 3. Chunking Comparison Checkpoints

When tuning retrieval, rebuild the index with alternate chunking and repeat the smoke procedure:

| Variant | Flags (example) | Hypothesis |
|---------|-----------------|------------|
| Default | (none) | Baseline overlap and granularity |
| Larger windows | `--lines-per-chunk 8 --line-overlap 2` | Better context, possibly worse precision |
| Tighter windows | `--lines-per-chunk 3 --line-overlap 0` | Sharper hits, possibly fragmented text |

**Pass criterion:** Same or better hit rate on `gil_queries.json` vs default, subjectively acceptable chunk readability.

## 4. Embedding Model Comparison Checkpoints

Rebuild with a second model and repeat smoke queries:

| Model | Build flag |
|-------|------------|
| Default | `--model all-MiniLM-L6-v2` |
| Alternative | `--model multi-qa-mpnet-base-cos-v1` |

Use a **separate** `--output-dir` per model to avoid overwriting artifacts.

**Pass criterion:** Improved ranking on failed queries from the default model, without regressing the majority of passes.

## 5. Automated Regression

- **Unit:** `tests/test_vector_index.py::test_faiss_search_matches_in_memory_cosine_search` ensures FAISS `IndexFlatIP` ordering matches in-memory inner product on normalized vectors (equivalent to cosine for unit vectors).
- **Retrieval service:** `tests/test_retrieval.py` covers bundle load, `retrieve` / `retrieve_from_disk`, config-driven model resolution (with mocked embeddings), citation dict shape, and invalid `top_k`.
- **RAG plumbing:** `tests/test_chat_prompt.py` (prompt/context assembly), `tests/test_openrouter.py` (mocked HTTP), `tests/test_webapp.py` (FastAPI routes with temp index and mocked embeddings).
- **Optional:** For debugging only, compare `sentence_transformers.util.semantic_search` on a tiny in-memory corpus against FAISS; not run in default CI to avoid model download.

## 6. Future Metrics (Deferred)

When the query set grows, consider:

- Recall@k per query
- MRR (mean reciprocal rank)
- nDCG with graded relevance labels

## 8. RAG And Web UI Smoke

Prerequisites: built index; web app running with `OPENROUTER_API_KEY` set (optional for retrieval-only mode).

**Retrieval bar (unchanged):** Run seed queries from `gil_queries.json` via `scripts/query_transcripts.py` or the shared `neural.retrieval` API; expect expected episode substring in top-5 `file=` lines.

**RAG bar (manual, v1):**

1. Ask each seed query in the web UI with generation enabled.
2. Pass if the answer either (a) clearly paraphrases only content visible in the listed evidence chunks, or (b) states that the corpus does not contain enough information—and the top evidence chunks are plausibly on-topic or empty.
3. Fail if the answer asserts facts not supported by any shown chunk.
4. Record failures in a local scratch log (not committed).

Optional future fields for `evals/gil_queries.json` or a sibling file: `min_citations`, `forbidden_claims`—deferred until automated checks are worth the maintenance.

## 9. References

| Topic | Location |
|-------|----------|
| Tech spec test cases | [gil-transcript-tech-spec.md](./gil-transcript-tech-spec.md#7-test-case-specifications) |
| Seed queries | [evals/gil_queries.json](../evals/gil_queries.json) |
| Query CLI | [scripts/query_transcripts.py](../scripts/query_transcripts.py) |
| Web app | [src/webapp/main.py](../src/webapp/main.py) |
| Chatbot ADR | [adr/002-citation-first-web-chatbot.md](./adr/002-citation-first-web-chatbot.md) |
