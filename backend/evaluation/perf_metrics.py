"""Latency and cost metrics for runtime and offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config import settings


@dataclass(frozen=True)
class ModelPricing:
    """USD pricing per 1M tokens."""

    input_per_1m: float
    output_per_1m: float


class CostCalculator:
    """Estimate cost from token usage and pricing tables.

    Pricing snapshot defaults are based on a February 14, 2026 internal baseline.
    Keep these values updated as provider pricing changes.
    """

    def __init__(self, pricing_by_model: dict[str, ModelPricing] | None = None):
        default = ModelPricing(
            input_per_1m=settings.price_input_per_1m_tokens,
            output_per_1m=settings.price_output_per_1m_tokens,
        )
        self.pricing_by_model = pricing_by_model or {
            settings.gemini_model: default,
            settings.eval_judge_model: default,
        }

    def estimate_cost_usd(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        pricing = self.pricing_by_model.get(model_name)
        if pricing is None:
            pricing = ModelPricing(
                input_per_1m=settings.price_input_per_1m_tokens,
                output_per_1m=settings.price_output_per_1m_tokens,
            )

        in_cost = (max(prompt_tokens, 0) / 1_000_000) * pricing.input_per_1m
        out_cost = (max(completion_tokens, 0) / 1_000_000) * pricing.output_per_1m
        return round(in_cost + out_cost, 8)


def percentile(values: Iterable[float], q: float) -> float:
    """Compute percentile with linear interpolation."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if q <= 0:
        return vals[0]
    if q >= 100:
        return vals[-1]

    pos = (len(vals) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(vals) - 1)
    frac = pos - low
    return vals[low] * (1 - frac) + vals[high] * frac


def summarize_latencies(latencies_ms: Iterable[float]) -> dict[str, float]:
    """Return p50/p95/p99 latency summary."""
    vals = [float(v) for v in latencies_ms]
    if not vals:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "avg_ms": 0.0}
    return {
        "p50_ms": round(percentile(vals, 50), 2),
        "p95_ms": round(percentile(vals, 95), 2),
        "p99_ms": round(percentile(vals, 99), 2),
        "avg_ms": round(sum(vals) / len(vals), 2),
    }
