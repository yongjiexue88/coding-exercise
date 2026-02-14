"""Schemas for evaluation datasets, per-case results, and run summaries."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class GoldChunkRef(BaseModel):
    """Chunk-level relevance target for retrieval evaluation."""

    source: str
    doc_index: Optional[int] = None
    paragraph_index: Optional[int] = None
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)


class EvalSample(BaseModel):
    """One dataset sample for evaluation runs."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_chunk_refs: list[GoldChunkRef] = Field(default_factory=list)
    reference_answer: Optional[str] = None
    tags: dict[str, Any] = Field(default_factory=dict)


class RetrievalMetrics(BaseModel):
    """Retrieval metrics for one sample."""

    k: int = 5
    precision_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    recall_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    mrr_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    ndcg_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieved_doc_ids: list[str] = Field(default_factory=list)


class AnswerMetrics(BaseModel):
    """Answer quality scores for one sample."""

    groundedness: float = Field(default=0.0, ge=0.0, le=1.0)
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination: bool = False
    uncertainty: bool = False


class PerfMetrics(BaseModel):
    """Performance telemetry for one sample."""

    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Per-sample full evaluation output."""

    run_id: str
    sample_id: str
    split: Literal["dev", "holdout", "unknown"] = "unknown"
    query: str
    answer: str = ""
    retrieval: RetrievalMetrics
    answer_metrics: AnswerMetrics
    perf: PerfMetrics
    judge_outputs: dict[str, Any] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)


class EvalRunSummary(BaseModel):
    """Aggregated run-level statistics."""

    run_id: str
    mode: Literal["retrieval_only", "full_rag_with_judges"]
    total_samples: int
    evaluated_samples: int
    retrieval: dict[str, float]
    answer: dict[str, float]
    perf: dict[str, float]
    gate_failures: list[str] = Field(default_factory=list)
