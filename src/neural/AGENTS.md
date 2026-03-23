# NEURAL PACKAGE

## OVERVIEW
Core retrieval library: transcript parsing, chunking, embeddings, FAISS helpers, retrieval orchestration, prompts, and OpenRouter integration.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Parse raw transcript files | `src/neural/corpus.py` | Timestamp regex, continuation-line merge, corpus loading |
| Tune chunk sizes | `src/neural/chunking.py` | `ChunkingConfig`, overlap rules, min-char gate |
| Encode text | `src/neural/embeddings.py` | Sentence-transformers boundary |
| Read/write FAISS bundle | `src/neural/vector_index.py` | Artifact persistence and exact search |
| Shared retrieval path | `src/neural/retrieval.py` | Used by CLI and web app |
| Build chat messages | `src/neural/chat_prompt.py` | RAG prompt surface |
| Call hosted model | `src/neural/openrouter.py` | Stdlib HTTP only |
| Metadata enrichment | `src/neural/glossary_extractor.py`, `src/neural/chunk_enrichment.py` | Deterministic + LLM rules |

## CONVENTIONS
- Value objects are usually frozen dataclasses; preserve immutability unless there is a strong reason not to.
- Retrieval helpers prefer small, composable functions over stateful service classes.
- Persisted artifacts stay JSON + FAISS files, not custom binary formats.
- Public package exports are curated through `src/neural/__init__.py`.

## ANTI-PATTERNS
- Do not change retrieval semantics in only one surface; CLI and web app should keep using the same retrieval path.
- Do not silently accept invalid `top_k`, bad embedding shapes, or missing artifacts; current modules raise explicit errors.
- Do not replace exact-search assumptions casually; roadmap treats ANN/reranking as follow-on work.
- Do not let metadata extraction invent names, schema fields, or uncontrolled vocab values.

## NOTES
- `src/neural/corpus.py` and `src/neural/chunking.py` encode the repo's core semi-structured text assumptions.
- `src/neural/vector_index.py` uses normalized similarity expectations; read docs before changing scoring/index type.
- `src/neural/openrouter.py` intentionally avoids extra dependencies; keep it lightweight.
