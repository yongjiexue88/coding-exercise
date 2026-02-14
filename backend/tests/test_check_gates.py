"""Tests for gate-check regression logic."""

import json
from pathlib import Path

import yaml

from evaluation import check_gates


def test_check_gates_fails_on_recall_drop(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()

    (reports / "retrieval_metrics.json").write_text(
        json.dumps({"summary": {"avg_recall_at_k": 0.40, "avg_mrr_at_k": 0.45}}),
        encoding="utf-8",
    )
    (reports / "answer_metrics.json").write_text(
        json.dumps({"summary": {"avg_groundedness": 0.9, "hallucination_rate": 0.05}}),
        encoding="utf-8",
    )
    (reports / "latency_cost_metrics.json").write_text(
        json.dumps({"summary": {"p95_ms": 1000, "avg_cost_usd": 0.001}}),
        encoding="utf-8",
    )

    gates_path = tmp_path / "gates.yaml"
    gates_path.write_text(
        yaml.safe_dump(
            {
                "retrieval": {
                    "recall_at_5_max_drop_pct_points": 3,
                    "mrr_at_5_max_drop_pct_points": 5,
                },
                "answer": {"groundedness_min": 0.85, "hallucination_rate_max": 0.08},
                "performance": {"p95_latency_ms_max": 3500, "avg_cost_usd_max": 0.015},
            }
        ),
        encoding="utf-8",
    )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "retrieval": {"avg_recall_at_k": 0.55, "avg_mrr_at_k": 0.50},
                "answer": {"avg_groundedness": 0.9, "hallucination_rate": 0.05},
                "perf": {"p95_ms": 2000, "avg_cost_usd": 0.01},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_gates.py",
            "--reports-dir",
            str(reports),
            "--gates",
            str(gates_path),
            "--baseline",
            str(baseline_path),
            "--scope",
            "retrieval",
        ],
    )

    assert check_gates.main() == 1


def test_check_gates_passes_when_thresholds_met(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()

    (reports / "retrieval_metrics.json").write_text(
        json.dumps({"summary": {"avg_recall_at_k": 0.55, "avg_mrr_at_k": 0.50}}),
        encoding="utf-8",
    )
    (reports / "answer_metrics.json").write_text(
        json.dumps({"summary": {"avg_groundedness": 0.9, "hallucination_rate": 0.05}}),
        encoding="utf-8",
    )
    (reports / "latency_cost_metrics.json").write_text(
        json.dumps({"summary": {"p95_ms": 1000, "avg_cost_usd": 0.001}}),
        encoding="utf-8",
    )

    gates_path = tmp_path / "gates.yaml"
    gates_path.write_text(
        yaml.safe_dump(
            {
                "retrieval": {
                    "recall_at_5_max_drop_pct_points": 3,
                    "mrr_at_5_max_drop_pct_points": 5,
                },
                "answer": {"groundedness_min": 0.85, "hallucination_rate_max": 0.08},
                "performance": {"p95_latency_ms_max": 3500, "avg_cost_usd_max": 0.015},
            }
        ),
        encoding="utf-8",
    )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "retrieval": {"avg_recall_at_k": 0.55, "avg_mrr_at_k": 0.50},
                "answer": {"avg_groundedness": 0.9, "hallucination_rate": 0.05},
                "perf": {"p95_ms": 2000, "avg_cost_usd": 0.01},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_gates.py",
            "--reports-dir",
            str(reports),
            "--gates",
            str(gates_path),
            "--baseline",
            str(baseline_path),
            "--scope",
            "full",
        ],
    )

    assert check_gates.main() == 0
