# ADR 001: Retrieval-First Gil Transcript NLP

## Status

Accepted

## Context

The repository currently focuses on transcript acquisition and storage. It contains a scraper, tests for HTML parsing, and a local corpus of transcript text files, but it does not contain a model-training stack, labels, or evaluation assets for supervised or generative NLP work.

The corpus is already suitable for semantic retrieval because transcript lines carry timestamps and episode provenance. It is not yet suitable for from-scratch generative model training because the data volume is modest, speaker labels are absent, and no gold evaluation set exists.

## Decision

The first NLP milestone will be a retrieval-first system built on:

- a local transcript corpus loader,
- deterministic timestamp-aware chunking,
- sentence-transformers embeddings,
- FAISS exact search with `IndexFlatIP`,
- and a CLI query surface.

The initial default embedding model is `all-MiniLM-L6-v2`, with model choice exposed as configuration rather than hardcoded behavior.

## Consequences

### Positive

- The repo gains a useful local NLP capability quickly.
- Timestamped evidence remains visible in every result.
- Retrieval quality can be tested before more ambitious modeling work.
- The package layout created for retrieval can support later reranking, clustering, and classification.

### Negative

- This milestone does not answer questions generatively on its own.
- Retrieval quality depends heavily on chunking strategy.
- Local index artifacts become part of the developer workflow.

### Deferred Follow-On Work

See [Phase Two Roadmap](../phase-two-roadmap.md) for detail. Summary:

- Cross-encoder reranking after baseline validation.
- Manual labeling for classification experiments.
- Clustering and topic discovery over chunk embeddings.
- A citation-first RAG layer if retrieval proves useful.

## Alternatives Considered

### Fine-Tune Or Train A Generator First

Rejected for milestone one because the current repo lacks labels, training infrastructure, and sufficient corpus scale for a strong first generative result.

### Supervised Classification First

Rejected for milestone one because there is no existing labeled dataset and the most reusable capability is semantic retrieval.

### Web App First

Rejected for milestone one because a CLI is enough to validate corpus parsing, chunking, embeddings, and ranking quality.

## References

| Topic | Location | Anchor |
|-------|----------|--------|
| Strategic blueprint | [Gil Transcript Retrieval Blueprint](../gil-transcript-blueprint.md#4-architectural-decisions) | `4-architectural-decisions` |
| Technical rules | [Gil Transcript Retrieval Tech Spec](../gil-transcript-tech-spec.md#3-chunking-rules) | `3-chunking-rules` |
| Success definition | [Blueprint success criteria](../gil-transcript-blueprint.md#3-success-criteria) | `3-success-criteria` |
