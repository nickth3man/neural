# Gil Transcript Retrieval Blueprint (Strategic)

## 1. Problem Statement

The repository already captures a meaningful local corpus of Gil's Arena transcripts, but it does not yet provide a usable NLP workflow over that data. The first milestone is to turn the transcript corpus into a local semantic retrieval system that answers free-text basketball questions with timestamped transcript evidence.

**Implementation Implication:** Build a retrieval-first package and CLI instead of a training pipeline or chat model.

## 2. First Milestone

The first deliverable is a local command-line workflow that:

- loads transcript files from `gil/transcripts`,
- parses timestamp-prefixed transcript lines,
- chunks the corpus into retrieval units with source provenance,
- embeds chunks with a configurable sentence-transformers model,
- indexes embeddings with a FAISS exact-search baseline,
- and returns top-k results with file and timestamp evidence.

**Implementation Implication:** Ship two scripts, one for index construction and one for querying, backed by importable package modules.

## 3. Success Criteria

The retrieval MVP is successful when all of the following are true:

- Index build completes against the local transcript directory without manual file editing.
- Query results include `episode_title`, `source_file`, `start_timestamp`, `end_timestamp`, and `chunk_text`.
- Retrieval behavior is covered by automated tests for parsing, chunking, and vector search.
- A small tracked evaluation set exists for realistic basketball prompts.
- The semantic retriever establishes a stronger baseline than naive filename or keyword-only lookup for the first milestone.

**Implementation Implication:** Add tracked evaluation fixtures and tests before considering second-wave modeling work.

## 4. Architectural Decisions

The first milestone uses a retrieval-first architecture rather than classifier training or generative model training.

- Corpus units are timestamped transcript chunks, not full episodes.
- The default embedder is `all-MiniLM-L6-v2`.
- The vector index is FAISS `IndexFlatIP` over normalized embeddings.
- A CLI is the only required product surface for **milestone one** (historical scope).
- Later reranking, clustering, and supervised labeling remain deferred beyond their roadmap sections.

**Implementation Implication:** Keep model choice configurable and keep the package modular so later retrieval quality experiments do not require rewriting the corpus layer.

## 5. Explicit Non-Goals (Milestone One)

The **first** milestone did not include:

- training a generative model from scratch,
- speaker diarization or speaker-aware retrieval,
- a web UI (this was deferred intentionally; see phase two),
- approximate ANN indexes,
- or supervised topic classification.

**Phase two (current extension):** A **citation-first local web chatbot** on top of the same retrieval stack is in scope per [ADR 002](./adr/002-citation-first-web-chatbot.md). It does not replace CLI validation or change the index artifact contract.

**Implementation Implication:** Avoid training loops and ANN complexity until the retrieval baseline is stable; web and RAG layers must surface transcript evidence, not hide it.

## 6. References

### Implementation Details Location
| Content Type | Location |
|--------------|----------|
| Retrieval implementation rules | [Gil Transcript Retrieval Tech Spec](./gil-transcript-tech-spec.md#1-scope-and-deliverable) |
| Anti-patterns | [Gil Transcript Retrieval Tech Spec](./gil-transcript-tech-spec.md#6-anti-patterns-do-not) |
| Test cases | [Gil Transcript Retrieval Tech Spec](./gil-transcript-tech-spec.md#7-test-case-specifications) |
| Error handling | [Gil Transcript Retrieval Tech Spec](./gil-transcript-tech-spec.md#8-error-handling-matrix) |
| Evaluation plan | [Gil Transcript Evaluation](./gil-transcript-evaluation.md) |
| Phase-two roadmap | [Phase Two Roadmap](./phase-two-roadmap.md) |
| Web chatbot (citation-first RAG) | [ADR 002](./adr/002-citation-first-web-chatbot.md) | — |

### Decision Records
| Topic | Location | Anchor |
|-------|----------|--------|
| Retrieval-first decision | [ADR 001](./adr/001-retrieval-first.md#decision) | `decision` |
| Citation-first web chatbot | [ADR 002](./adr/002-citation-first-web-chatbot.md) | — |

### Repository References
| Topic | Location | Anchor |
|-------|----------|--------|
| Scraper output source | [Scraper script](../scripts/scrape_podscripts.py) | `scripts/scrape_podscripts.py` |
| Transcript corpus location | [Transcript directory](../gil/transcripts) | `gil/transcripts` |
| Retrieval scripts | [Tech Spec implementation surface](./gil-transcript-tech-spec.md#5-cli-and-artifact-surface) | `5-cli-and-artifact-surface` |

*This document is strategic. Implementation rules, tests, and failure handling live in the technical specification.*
