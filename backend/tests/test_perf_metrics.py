"""Tests for cost and latency helpers."""

from evaluation.perf_metrics import CostCalculator, ModelPricing, percentile, summarize_latencies


def test_cost_calculator_uses_model_pricing():
    calc = CostCalculator(
        pricing_by_model={
            "model-a": ModelPricing(input_per_1m=1.0, output_per_1m=2.0),
        }
    )
    cost = calc.estimate_cost_usd("model-a", prompt_tokens=500_000, completion_tokens=250_000)
    assert cost == 1.0


def test_percentile_and_summary():
    vals = [10, 20, 30, 40, 50]
    assert percentile(vals, 50) == 30
    summary = summarize_latencies(vals)
    assert summary["p95_ms"] >= summary["p50_ms"]
    assert summary["avg_ms"] == 30.0
