"""Evaluation runner for retrieval-only and full-RAG judge-based evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from sqlmodel import Session, select

from database import engine
from evaluation.judges import GroundednessJudge, QualityJudge
from evaluation.perf_metrics import CostCalculator, summarize_latencies
from evaluation.retrieval_metrics import compute_retrieval_metrics
from evaluation.schemas import (
    AnswerMetrics,
    EvalResult,
    EvalRunSummary,
    EvalSample,
    PerfMetrics,
)
from models_observability import EvalCaseResult, EvalRun
from services.embedding import EmbeddingService
from services.llm import LLMService
from services.vector_store import VectorStoreService


DATASET_DEFAULT = Path(__file__).parent / "datasets" / "eval_v1.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"


def _load_dataset(path: Path, limit: int | None = None) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(EvalSample.model_validate_json(line))
            if limit and len(samples) >= limit:
                break
    return samples


def _persist_eval_run_start(run_id: str, mode: str, dataset_path: str, total_samples: int) -> None:
    try:
        with Session(engine) as session:
            session.add(
                EvalRun(
                    run_id=run_id,
                    mode=mode,
                    status="running",
                    dataset_path=dataset_path,
                    total_samples=total_samples,
                    summary_json={},
                )
            )
            session.commit()
    except Exception:
        return


def _persist_case_result(result: EvalResult) -> None:
    try:
        with Session(engine) as session:
            session.add(
                EvalCaseResult(
                    run_id=result.run_id,
                    sample_id=result.sample_id,
                    split=result.split,
                    query_text=result.query,
                    retrieval_metrics_json=result.retrieval.model_dump(),
                    answer_metrics_json=result.answer_metrics.model_dump(),
                    perf_metrics_json=result.perf.model_dump(),
                    judge_outputs_json=result.judge_outputs,
                    failure_reasons_json=result.failure_reasons,
                )
            )
            session.commit()
    except Exception:
        return


def _persist_eval_run_complete(run_id: str, summary: dict[str, Any]) -> None:
    try:
        with Session(engine) as session:
            run = session.exec(select(EvalRun).where(EvalRun.run_id == run_id)).first()
            if not run:
                return
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.summary_json = summary
            session.add(run)
            session.commit()
    except Exception:
        return


def _split_from_tags(tags: dict[str, Any]) -> Literal["dev", "holdout", "unknown"]:
    split = str(tags.get("split", "unknown")).lower()
    if split in {"dev", "holdout"}:
        return split  # type: ignore[return-value]
    return "unknown"


async def run_evaluation(
    mode: Literal["retrieval_only", "full_rag_with_judges"] = "full_rag_with_judges",
    dataset_path: Path = DATASET_DEFAULT,
    limit: int | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run evaluation and write report artifacts."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    samples = _load_dataset(dataset_path, limit=limit)
    if not samples:
        raise ValueError("Evaluation dataset is empty.")

    run_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    _persist_eval_run_start(run_id, mode, str(dataset_path), len(samples))

    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()
    llm_service = LLMService()
    groundedness_judge = GroundednessJudge(llm_service=llm_service)
    quality_judge = QualityJudge(llm_service=llm_service)
    cost_calculator = CostCalculator()

    results: list[EvalResult] = []
    failed_examples: list[dict[str, Any]] = []

    for sample in samples:
        case_started = time.time()
        retrieval_started = time.time()

        query_embedding = embedding_service.embed_query(sample.query)
        search_results = vector_store.search(query_embedding, top_k=top_k)

        retrieved_docs = search_results.get("documents", [[]])[0] if search_results.get("documents") else []
        retrieved_metas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []
        retrieved_sources = [meta.get("source", "") for meta in retrieved_metas]

        retrieval_latency_ms = (time.time() - retrieval_started) * 1000

        retrieval = compute_retrieval_metrics(
            retrieved_doc_ids=retrieved_sources,
            gold_doc_ids=sample.gold_doc_ids,
            gold_chunk_refs=sample.gold_chunk_refs,
            k=top_k,
        )

        answer_text = ""
        answer_metrics = AnswerMetrics()
        judge_outputs: dict[str, Any] = {}

        llm_prompt_tokens = 0
        llm_completion_tokens = 0
        llm_total_tokens = 0
        llm_latency_ms = 0.0
        llm_cost_usd = 0.0

        context_docs = [
            {"content": doc, "source": meta.get("source", "Unknown")}
            for doc, meta in zip(retrieved_docs, retrieved_metas)
        ]

        if mode == "full_rag_with_judges":
            llm_started = time.time()
            if hasattr(llm_service, "generate_with_metadata"):
                answer_text, usage = await llm_service.generate_with_metadata(
                    query=sample.query,
                    context_docs=context_docs,
                    route="rag_simple",
                )
            else:
                answer_text = await llm_service.generate(
                    query=sample.query,
                    context_docs=context_docs,
                    route="rag_simple",
                )
                usage = {}

            llm_latency_ms = float(usage.get("latency_ms", 0.0)) or ((time.time() - llm_started) * 1000)
            llm_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            llm_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            llm_total_tokens = int(usage.get("total_tokens", 0) or 0)
            llm_cost_usd = cost_calculator.estimate_cost_usd(
                model_name=str(usage.get("model_name", llm_service.current_model_name)),
                prompt_tokens=llm_prompt_tokens,
                completion_tokens=llm_completion_tokens,
            )

            grounded = await groundedness_judge.evaluate(
                query=sample.query,
                answer=answer_text,
                context_docs=context_docs,
                reference_answer=sample.reference_answer,
            )
            quality = await quality_judge.evaluate(
                query=sample.query,
                answer=answer_text,
                context_docs=context_docs,
                reference_answer=sample.reference_answer,
            )

            judge_outputs = {
                "groundedness": grounded.model_dump(),
                "quality": quality.model_dump(),
            }

            hallucination = grounded.output.error_type == "hallucination"
            uncertainty = grounded.output.uncertainty or quality.output.uncertainty
            answer_metrics = AnswerMetrics(
                groundedness=grounded.output.score,
                quality=quality.output.score,
                hallucination=hallucination,
                uncertainty=uncertainty,
            )

        total_latency_ms = (time.time() - case_started) * 1000
        perf = PerfMetrics(
            total_latency_ms=round(total_latency_ms, 2),
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            llm_latency_ms=round(llm_latency_ms, 2),
            prompt_tokens=llm_prompt_tokens,
            completion_tokens=llm_completion_tokens,
            total_tokens=llm_total_tokens,
            cost_usd=llm_cost_usd,
            stage_latency_ms={
                "retrieval": round(retrieval_latency_ms, 2),
                "llm": round(llm_latency_ms, 2),
            },
        )

        failure_reasons: list[str] = []
        if retrieval.recall_at_k < 1.0 and sample.gold_doc_ids:
            failure_reasons.append("retrieval_miss")
        if mode == "full_rag_with_judges" and answer_metrics.hallucination:
            failure_reasons.append("hallucination")

        result = EvalResult(
            run_id=run_id,
            sample_id=sample.id,
            split=_split_from_tags(sample.tags),
            query=sample.query,
            answer=answer_text,
            retrieval=retrieval,
            answer_metrics=answer_metrics,
            perf=perf,
            judge_outputs=judge_outputs,
            failure_reasons=failure_reasons,
        )

        results.append(result)
        _persist_case_result(result)

        if failure_reasons:
            failed_examples.append(
                {
                    "run_id": run_id,
                    "sample_id": sample.id,
                    "query": sample.query,
                    "failure_reasons": failure_reasons,
                    "retrieval": retrieval.model_dump(),
                    "answer_metrics": answer_metrics.model_dump(),
                }
            )

    retrieval_summary = {
        "avg_precision_at_k": round(sum(r.retrieval.precision_at_k for r in results) / len(results), 4),
        "avg_recall_at_k": round(sum(r.retrieval.recall_at_k for r in results) / len(results), 4),
        "avg_mrr_at_k": round(sum(r.retrieval.mrr_at_k for r in results) / len(results), 4),
        "avg_ndcg_at_k": round(sum(r.retrieval.ndcg_at_k for r in results) / len(results), 4),
        "k": top_k,
    }

    answer_summary = {
        "avg_groundedness": round(sum(r.answer_metrics.groundedness for r in results) / len(results), 4),
        "avg_quality": round(sum(r.answer_metrics.quality for r in results) / len(results), 4),
        "hallucination_rate": round(
            sum(1 for r in results if r.answer_metrics.hallucination) / len(results),
            4,
        ),
        "uncertainty_rate": round(
            sum(1 for r in results if r.answer_metrics.uncertainty) / len(results),
            4,
        ),
    }

    latencies = [r.perf.total_latency_ms for r in results]
    perf_summary = {
        **summarize_latencies(latencies),
        "avg_cost_usd": round(sum(r.perf.cost_usd for r in results) / len(results), 8),
        "avg_prompt_tokens": round(sum(r.perf.prompt_tokens for r in results) / len(results), 2),
        "avg_completion_tokens": round(sum(r.perf.completion_tokens for r in results) / len(results), 2),
        "avg_total_tokens": round(sum(r.perf.total_tokens for r in results) / len(results), 2),
    }

    summary_model = EvalRunSummary(
        run_id=run_id,
        mode=mode,
        total_samples=len(samples),
        evaluated_samples=len(results),
        retrieval=retrieval_summary,
        answer=answer_summary,
        perf=perf_summary,
        gate_failures=[],
    )

    retrieval_report = {
        "run_id": run_id,
        "mode": mode,
        "summary": retrieval_summary,
        "results": [
            {
                "sample_id": r.sample_id,
                "split": r.split,
                "query": r.query,
                "retrieval": r.retrieval.model_dump(),
            }
            for r in results
        ],
    }
    answer_report = {
        "run_id": run_id,
        "mode": mode,
        "summary": answer_summary,
        "results": [
            {
                "sample_id": r.sample_id,
                "split": r.split,
                "query": r.query,
                "answer_metrics": r.answer_metrics.model_dump(),
                "judge_outputs": r.judge_outputs,
            }
            for r in results
        ],
    }
    perf_report = {
        "run_id": run_id,
        "mode": mode,
        "summary": perf_summary,
        "results": [
            {
                "sample_id": r.sample_id,
                "split": r.split,
                "query": r.query,
                "perf": r.perf.model_dump(),
            }
            for r in results
        ],
    }

    (REPORTS_DIR / "retrieval_metrics.json").write_text(json.dumps(retrieval_report, indent=2), encoding="utf-8")
    (REPORTS_DIR / "answer_metrics.json").write_text(json.dumps(answer_report, indent=2), encoding="utf-8")
    (REPORTS_DIR / "latency_cost_metrics.json").write_text(json.dumps(perf_report, indent=2), encoding="utf-8")

    failed_path = REPORTS_DIR / "examples_failed.jsonl"
    with failed_path.open("w", encoding="utf-8") as f:
        for row in failed_examples:
            f.write(json.dumps(row) + "\n")

    final_summary = summary_model.model_dump()
    _persist_eval_run_complete(run_id, final_summary)

    return {
        "run_id": run_id,
        "mode": mode,
        "dataset": str(dataset_path),
        "total_samples": len(samples),
        "reports_dir": str(REPORTS_DIR),
        "summary": final_summary,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for module execution."""
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
