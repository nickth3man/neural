#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neural.evaluation import evaluate_retrieval, load_eval_cases
from neural.metadata_index import RetrievalFilters, load_metadata_index
from neural.reranking import RerankerConfig
from neural.retrieval import load_retrieval_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality from tracked seed queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data/transcript_index"))
    parser.add_argument("--evals", type=Path, default=Path("evals/gil_queries.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--episode-type", type=str, default=None)
    parser.add_argument("--guest", type=str, default=None)
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--team", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--source-file", type=str, default=None)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--reranker-model", type=str, default=RerankerConfig().model_name)
    parser.add_argument("--rerank-top-n", type=int, default=RerankerConfig().top_n)
    parser.add_argument("--hybrid", action="store_true", help="BM25 + dense RRF fusion")
    parser.add_argument("--hybrid-lexical-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_retrieval_bundle(args.index_dir)
    eval_cases = load_eval_cases(args.evals)
    metadata_index = load_metadata_index(args.metadata_dir) if args.metadata_dir else None
    filters = RetrievalFilters(
        episode_type=args.episode_type,
        guest_name=args.guest,
        speaker=args.speaker,
        team=args.team,
        topic=args.topic,
        source_file=args.source_file,
    )
    reranker = None
    if args.rerank:
        reranker = RerankerConfig(model_name=args.reranker_model, top_n=args.rerank_top_n)

    summary = evaluate_retrieval(
        bundle,
        eval_cases,
        top_k=args.top_k,
        metadata_index=metadata_index,
        filters=filters,
        reranker=reranker,
        hybrid=args.hybrid,
        hybrid_lexical_k=args.hybrid_lexical_k,
        rrf_k=args.rrf_k,
    )

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
        return

    print(f"Queries:      {summary.total_queries}")
    print(f"Hit rate@{args.top_k}: {summary.hit_rate_at_k:.3f}")
    print(f"MRR:          {summary.mrr:.3f}")
    failures = [result for result in summary.results if not result.matched]
    if failures:
        print("Failures:")
        for result in failures:
            actual = ", ".join(result.result_files) or "no results"
            print(f"- {result.query} -> expected {result.expected_episode_substring}; got {actual}")


if __name__ == "__main__":
    main()
