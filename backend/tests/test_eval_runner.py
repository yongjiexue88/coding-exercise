"""Integration-like tests for evaluation runner artifact outputs."""

import json
from pathlib import Path

import pytest

import evaluation.runner as runner
from evaluation.judges import JudgeOutput, JudgeResult


class FakeEmbeddingService:
    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3]


class FakeVectorStoreService:
    def search(self, query_embedding, top_k=5):
        return {
            "documents": [["Doc text A", "Doc text B"]],
            "metadatas": [[{"source": "docA.md"}, {"source": "docB.md"}]],
            "distances": [[0.1, 0.2]],
        }


class FakeLLMService:
    current_model_name = "fake-model"

    async def generate_with_metadata(self, **kwargs):
        return "fake answer", {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "latency_ms": 10.0,
            "model_name": "fake-model",
        }


class FakeJudge:
    def __init__(self, name: str):
        self.name = name

    async def evaluate(self, **kwargs):
        output = JudgeOutput(
            score=0.9,
            decision="pass",
            error_type="none",
            citations=[{"source": "docA.md", "snippet": "Doc text A"}],
            uncertainty=False,
            reasons=["looks good"],
        )
        return JudgeResult(name=self.name, output=output, usage={})


@pytest.mark.asyncio
async def test_runner_writes_reports(tmp_path, monkeypatch):
    dataset_path = tmp_path / "eval.jsonl"
    rows = [
        {
            "id": "1",
            "query": "question 1",
            "gold_doc_ids": ["docA.md"],
            "gold_chunk_refs": [{"source": "docA.md", "relevance": 1.0}],
            "reference_answer": "answer",
            "tags": {"split": "dev"},
        },
        {
            "id": "2",
            "query": "question 2",
            "gold_doc_ids": ["docB.md"],
            "gold_chunk_refs": [{"source": "docB.md", "relevance": 1.0}],
            "reference_answer": "answer",
            "tags": {"split": "holdout"},
        },
    ]
    dataset_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    reports_dir = tmp_path / "reports"

    monkeypatch.setattr(runner, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(runner, "VectorStoreService", FakeVectorStoreService)
    monkeypatch.setattr(runner, "LLMService", FakeLLMService)
    monkeypatch.setattr(runner, "GroundednessJudge", lambda llm_service: FakeJudge("groundedness"))
    monkeypatch.setattr(runner, "QualityJudge", lambda llm_service: FakeJudge("quality"))
    monkeypatch.setattr(runner, "REPORTS_DIR", reports_dir)

    output = await runner.run_evaluation(
        mode="full_rag_with_judges",
        dataset_path=dataset_path,
        limit=None,
        top_k=2,
    )

    assert output["total_samples"] == 2
    assert (reports_dir / "retrieval_metrics.json").exists()
    assert (reports_dir / "answer_metrics.json").exists()
    assert (reports_dir / "latency_cost_metrics.json").exists()
    assert (reports_dir / "examples_failed.jsonl").exists()


@pytest.mark.asyncio
async def test_legacy_cli_entrypoint_behavior(tmp_path, monkeypatch):
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "1",
                "query": "question 1",
                "gold_doc_ids": ["docA.md"],
                "gold_chunk_refs": [{"source": "docA.md", "relevance": 1.0}],
                "reference_answer": "answer",
                "tags": {"split": "dev"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reports_dir = tmp_path / "reports"

    monkeypatch.setattr(runner, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(runner, "VectorStoreService", FakeVectorStoreService)
    monkeypatch.setattr(runner, "LLMService", FakeLLMService)
    monkeypatch.setattr(runner, "GroundednessJudge", lambda llm_service: FakeJudge("groundedness"))
    monkeypatch.setattr(runner, "QualityJudge", lambda llm_service: FakeJudge("quality"))
    monkeypatch.setattr(runner, "REPORTS_DIR", reports_dir)

    output = await runner.run_evaluation(mode="retrieval_only", dataset_path=dataset_path, top_k=2)
    assert output["mode"] == "retrieval_only"
