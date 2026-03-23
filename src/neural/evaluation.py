from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from neural.metadata_index import MetadataIndex, RetrievalFilters
from neural.reranking import RerankerConfig
from neural.retrieval import RetrievalBundle, retrieve


@dataclass(frozen=True, slots=True)
class EvalCase:
    query: str
    expected_episode_substring: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    query: str
    expected_episode_substring: str
    matched: bool
    reciprocal_rank: float
    result_files: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class EvalSummary:
    total_queries: int
    hit_rate_at_k: float
    mrr: float
    results: tuple[EvalCaseResult, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["results"] = [asdict(result) for result in self.results]
        return data


def load_eval_cases(eval_path: Path) -> list[EvalCase]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            query=str(item["query"]),
            expected_episode_substring=str(item["expected_episode_substring"]),
            notes=str(item.get("notes", "")),
        )
        for item in data
    ]


def evaluate_retrieval(
    bundle: RetrievalBundle,
    eval_cases: list[EvalCase],
    *,
    top_k: int = 5,
    metadata_index: MetadataIndex | None = None,
    filters: RetrievalFilters | None = None,
    reranker: RerankerConfig | None = None,
    hybrid: bool = False,
    hybrid_lexical_k: int = 20,
    rrf_k: int = 60,
) -> EvalSummary:
    results: list[EvalCaseResult] = []
    reciprocal_rank_total = 0.0
    hits = 0

    for case in eval_cases:
        query_results = retrieve(
            bundle,
            case.query,
            top_k=top_k,
            metadata_index=metadata_index,
            filters=filters,
            reranker=reranker,
            hybrid=hybrid,
            hybrid_lexical_k=hybrid_lexical_k,
            rrf_k=rrf_k,
        )
        filenames = tuple(result.chunk.source_file for result in query_results)
        expected = case.expected_episode_substring.casefold()
        rank = 0
        for index, filename in enumerate(filenames, start=1):
            if expected in filename.casefold():
                rank = index
                break
        matched = rank > 0
        if matched:
            hits += 1
            reciprocal_rank_total += 1.0 / rank
        results.append(
            EvalCaseResult(
                query=case.query,
                expected_episode_substring=case.expected_episode_substring,
                matched=matched,
                reciprocal_rank=(1.0 / rank) if rank else 0.0,
                result_files=filenames,
                notes=case.notes,
            )
        )

    total = len(eval_cases)
    hit_rate = hits / total if total else 0.0
    mrr = reciprocal_rank_total / total if total else 0.0
    return EvalSummary(
        total_queries=total,
        hit_rate_at_k=hit_rate,
        mrr=mrr,
        results=tuple(results),
    )
