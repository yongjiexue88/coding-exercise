"""SQLModel tables for runtime observability and evaluation history."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Index
from sqlmodel import JSON, Column, Field, SQLModel


class QueryObservation(SQLModel, table=True):
    """Per-query runtime telemetry and policy decisions."""

    __tablename__ = "query_observation"

    id: Optional[int] = Field(default=None, primary_key=True)
    trace_id: str = Field(index=True)
    query_text: str
    route: str = Field(default="rag_simple", index=True)
    validator_decision: str = Field(default="pass", index=True)
    query_time_ms: float = Field(default=0.0)
    total_tokens: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)
    stage_latency_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    token_usage_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    eval_flags_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    validator_scores_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class EvalRun(SQLModel, table=True):
    """Top-level record for each offline evaluation run."""

    __tablename__ = "eval_run"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, unique=True)
    mode: str = Field(index=True)
    status: str = Field(default="running", index=True)
    dataset_path: str
    total_samples: int = Field(default=0)
    summary_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    started_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    completed_at: Optional[datetime] = Field(default=None)


class EvalCaseResult(SQLModel, table=True):
    """Per-sample evaluation outputs for debugging and audits."""

    __tablename__ = "eval_case_result"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    sample_id: str = Field(index=True)
    split: str = Field(default="dev", index=True)
    query_text: str
    retrieval_metrics_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    answer_metrics_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    perf_metrics_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    judge_outputs_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    failure_reasons_json: list = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


Index("idx_query_observation_created_route", QueryObservation.created_at, QueryObservation.route)
Index("idx_eval_case_run_split", EvalCaseResult.run_id, EvalCaseResult.split)
