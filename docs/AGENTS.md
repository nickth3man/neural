# DOCS

## OVERVIEW
Design record for the repo: blueprint, tech spec, evaluation guidance, ADRs, and phase-two follow-on work.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Strategic intent | `docs/gil-transcript-blueprint.md` | Milestone scope and non-goals |
| Implementation rules | `docs/gil-transcript-tech-spec.md` | Retrieval/chunking/indexing constraints |
| Retrieval-first decision | `docs/adr/001-retrieval-first.md` | Rejected alternatives included |
| Citation-first chat decision | `docs/adr/002-citation-first-web-chatbot.md` | Web app scope and non-goals |
| Follow-on work | `docs/phase-two-roadmap.md` | Reranking, clustering, supervised work, RAG |
| Eval expectations | `docs/gil-transcript-evaluation.md` | Smoke checks and quality bar |

## CONVENTIONS
- Treat docs here as live architectural constraints, not optional prose.
- Prefer updating the relevant ADR/spec when changing retrieval assumptions or scope boundaries.
- Keep milestone boundaries explicit; several docs separate MVP requirements from deferred work.

## ANTI-PATTERNS
- Do not implement roadmap ideas as if they are already baseline requirements.
- Do not weaken the retrieval-first or citation-first guarantees without updating the ADRs/specs.
- Do not treat phase-two items like reranking, auth, or persistent history as already in scope for v1.

## NOTES
- The strongest explicit anti-pattern list lives in `docs/gil-transcript-tech-spec.md`.
- `docs/phase-two-roadmap.md` includes a Windows-specific `DataLoader` guard note that matters if supervised work is added later.

## WORKFLOW
- Update the relevant ADR/spec in the same change when behavior or scope moves.
- Use `docs/adr/` for architectural decisions and the roadmap for deferred work, not the other way around.
