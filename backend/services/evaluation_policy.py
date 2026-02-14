"""Runtime policy overrides derived from recent evaluation outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, select

from config import settings
from database import engine
from models_observability import EvalRun


@dataclass
class RuntimeEvalPolicy:
    """Policy values consumed by RAG runtime before executing a query."""

    force_strict_validation: bool = False
    min_top_k: int = 1
    reduce_risky_routes: bool = False
    reason: str = "default"

    def as_flags(self) -> dict[str, Any]:
        return {
            "force_strict_validation": self.force_strict_validation,
            "min_top_k": self.min_top_k,
            "reduce_risky_routes": self.reduce_risky_routes,
            "policy_reason": self.reason,
        }


class EvaluationPolicyService:
    """Build runtime policy from latest stored evaluation summaries."""

    def get_runtime_policy(self) -> RuntimeEvalPolicy:
        if not settings.eval_feedback_enabled:
            return RuntimeEvalPolicy(reason="disabled")

        try:
            with Session(engine) as session:
                stmt = select(EvalRun).where(EvalRun.status == "completed").order_by(EvalRun.started_at.desc())
                latest = session.exec(stmt).first()
        except Exception:
            return RuntimeEvalPolicy(reason="eval_db_unavailable")

        if not latest:
            return RuntimeEvalPolicy(reason="no_eval_history")

        summary = latest.summary_json or {}
        answer_summary = summary.get("answer", {}) if isinstance(summary, dict) else {}

        groundedness = float(answer_summary.get("avg_groundedness", 1.0))
        hallucination_rate = float(answer_summary.get("hallucination_rate", 0.0))

        degrade_quality = (
            groundedness < settings.eval_feedback_groundedness_threshold
            or hallucination_rate > settings.eval_feedback_hallucination_threshold
        )

        if degrade_quality:
            return RuntimeEvalPolicy(
                force_strict_validation=True,
                min_top_k=max(settings.eval_feedback_min_top_k, settings.top_k),
                reduce_risky_routes=True,
                reason=(
                    f"quality_guard(groundedness={groundedness:.3f},"
                    f"hallucination_rate={hallucination_rate:.3f})"
                ),
            )

        return RuntimeEvalPolicy(reason="healthy")
