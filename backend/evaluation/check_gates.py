"""Gate check script for CI regression enforcement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check evaluation gates")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(__file__).parent / "reports",
    )
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path(__file__).parent / "gates.yaml",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).parent / "baselines" / "main.json",
    )
    parser.add_argument(
        "--scope",
        choices=["retrieval", "full"],
        default="full",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    with args.gates.open("r", encoding="utf-8") as f:
        gates = yaml.safe_load(f) or {}

    baseline = _load_json(args.baseline)
    retrieval_report = _load_json(args.reports_dir / "retrieval_metrics.json")
    answer_report = _load_json(args.reports_dir / "answer_metrics.json")
    perf_report = _load_json(args.reports_dir / "latency_cost_metrics.json")

    failures: list[str] = []

    retrieval_summary = retrieval_report.get("summary", {})
    baseline_retrieval = baseline.get("retrieval", {})
    recall_current = float(retrieval_summary.get("avg_recall_at_k", 0.0))
    mrr_current = float(retrieval_summary.get("avg_mrr_at_k", 0.0))
    recall_baseline = float(baseline_retrieval.get("avg_recall_at_k", 0.0))
    mrr_baseline = float(baseline_retrieval.get("avg_mrr_at_k", 0.0))

    recall_drop_pp = (recall_baseline - recall_current) * 100
    mrr_drop_pp = (mrr_baseline - mrr_current) * 100

    max_recall_drop = float(gates.get("retrieval", {}).get("recall_at_5_max_drop_pct_points", 999.0))
    max_mrr_drop = float(gates.get("retrieval", {}).get("mrr_at_5_max_drop_pct_points", 999.0))

    if recall_drop_pp > max_recall_drop:
        failures.append(
            f"recall@5 drop too large: drop={recall_drop_pp:.2f}pp threshold={max_recall_drop:.2f}pp"
        )

    if mrr_drop_pp > max_mrr_drop:
        failures.append(
            f"mrr@5 drop too large: drop={mrr_drop_pp:.2f}pp threshold={max_mrr_drop:.2f}pp"
        )

    if args.scope == "full":
        answer_summary = answer_report.get("summary", {})
        perf_summary = perf_report.get("summary", {})

        groundedness = float(answer_summary.get("avg_groundedness", 0.0))
        hallucination_rate = float(answer_summary.get("hallucination_rate", 1.0))

        min_groundedness = float(gates.get("answer", {}).get("groundedness_min", 0.0))
        max_hallucination_rate = float(gates.get("answer", {}).get("hallucination_rate_max", 1.0))

        if groundedness < min_groundedness:
            failures.append(
                f"groundedness below threshold: current={groundedness:.4f} min={min_groundedness:.4f}"
            )

        if hallucination_rate > max_hallucination_rate:
            failures.append(
                f"hallucination rate above threshold: current={hallucination_rate:.4f} max={max_hallucination_rate:.4f}"
            )

        p95_latency = float(perf_summary.get("p95_ms", 0.0))
        avg_cost = float(perf_summary.get("avg_cost_usd", 0.0))

        max_p95 = float(gates.get("performance", {}).get("p95_latency_ms_max", 1e9))
        max_cost = float(gates.get("performance", {}).get("avg_cost_usd_max", 1e9))

        if p95_latency > max_p95:
            failures.append(
                f"p95 latency above threshold: current={p95_latency:.2f}ms max={max_p95:.2f}ms"
            )

        if avg_cost > max_cost:
            failures.append(
                f"avg cost above threshold: current={avg_cost:.6f} max={max_cost:.6f}"
            )

    if failures:
        print("GATE CHECK: FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    print("GATE CHECK: PASSED")
    print(
        json.dumps(
            {
                "scope": args.scope,
                "retrieval": retrieval_summary,
                "answer": answer_report.get("summary", {}),
                "perf": perf_report.get("summary", {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
