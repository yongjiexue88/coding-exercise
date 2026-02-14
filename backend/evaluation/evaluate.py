"""CLI entrypoint for evaluation runner."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evaluation.runner import DATASET_DEFAULT, run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument(
        "--mode",
        choices=["retrieval_only", "full_rag_with_judges"],
        default="full_rag_with_judges",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
        help="Path to JSONL evaluation dataset.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval depth.")
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    output = await run_evaluation(
        mode=args.mode,
        dataset_path=args.dataset,
        limit=args.limit,
        top_k=args.top_k,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
