"""Evaluation framework for the RAG system."""

import json
import time
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.embedding import EmbeddingService
from services.vector_store import VectorStoreService
from services.llm import LLMService
from evaluation.metrics import (
    context_precision,
    context_recall_at_k,
    keyword_coverage,
    answer_length_score,
    faithfulness_heuristic,
)


TEST_QUERIES_PATH = Path(__file__).parent / "test_queries.json"


async def run_evaluation(top_k: int = 3) -> dict:
    """Run the full evaluation pipeline."""

    # Load test queries
    with open(TEST_QUERIES_PATH) as f:
        test_queries = json.load(f)

    print(f"\n🔍 Running evaluation with {len(test_queries)} queries...\n")

    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()
    llm_service = LLMService()

    results = []

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_source = test_case["expected_source"]
        expected_keywords = test_case.get("expected_keywords", [])

        print(f"  [{i}/{len(test_queries)}] {query[:60]}...")

        start_time = time.time()

        # Embed & retrieve
        query_embedding = embedding_service.embed_query(query)
        search_results = vector_store.search(query_embedding, top_k=top_k)

        retrieved_sources = []
        retrieved_docs = []
        if search_results["documents"] and search_results["documents"][0]:
            for j, doc in enumerate(search_results["documents"][0]):
                meta = search_results["metadatas"][0][j]
                retrieved_sources.append(meta.get("source", ""))
                retrieved_docs.append(doc)

        # Generate answer
        context_docs = [
            {"content": doc, "source": src}
            for doc, src in zip(retrieved_docs, retrieved_sources)
        ]

        try:
            answer = await llm_service.generate(query, context_docs)
        except Exception as e:
            answer = f"[ERROR: {e}]"

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # Calculate metrics
        cp = context_precision(retrieved_sources, expected_source)
        cr = context_recall_at_k(retrieved_sources, expected_source)
        kc = keyword_coverage(answer, expected_keywords)
        als = answer_length_score(answer)
        fh = faithfulness_heuristic(answer, retrieved_docs)

        result = {
            "query": query,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "metrics": {
                "context_precision": cp,
                "context_recall_at_k": cr,
                "keyword_coverage": kc,
                "answer_length_score": als,
                "faithfulness": fh,
            },
            "response_time_ms": elapsed_ms,
        }
        results.append(result)

    # Aggregate metrics
    metrics_keys = ["context_precision", "context_recall_at_k", "keyword_coverage",
                    "answer_length_score", "faithfulness"]
    averages = {}
    for key in metrics_keys:
        values = [r["metrics"][key] for r in results]
        averages[key] = round(sum(values) / len(values), 4)

    avg_time = round(sum(r["response_time_ms"] for r in results) / len(results), 2)

    # Print summary table
    print("\n" + "=" * 65)
    print("                    EVALUATION RESULTS")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Score':>10}")
    print("-" * 65)
    for key, value in averages.items():
        label = key.replace("_", " ").title()
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {label:<30} {bar} {value:.2%}")
    print("-" * 65)
    print(f"  {'Avg Response Time':<30} {avg_time:>10.0f} ms")
    print(f"  {'Queries Evaluated':<30} {len(results):>10}")
    print("=" * 65)

    # Save full results
    output = {
        "summary": averages,
        "avg_response_time_ms": avg_time,
        "total_queries": len(results),
        "results": results,
    }

    output_path = Path(__file__).parent.parent / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📊 Full results saved to {output_path}")

    return output


if __name__ == "__main__":
    asyncio.run(run_evaluation())
