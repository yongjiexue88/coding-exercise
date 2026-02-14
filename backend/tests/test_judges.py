"""Tests for LLM judge integration contracts."""

import pytest

from evaluation.judges import GroundednessJudge, JudgeOutput, QualityJudge


class FakeJudgeLLMService:
    def __init__(self, payload: JudgeOutput):
        self.payload = payload

    def generate_structured_with_metadata(self, **kwargs):
        return self.payload, {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "latency_ms": 12.3,
            "model_name": "fake-judge-model",
        }


@pytest.mark.asyncio
async def test_groundedness_judge_returns_contract():
    fake_output = JudgeOutput(
        score=0.92,
        decision="pass",
        error_type="none",
        citations=[{"source": "doc.md", "snippet": "supported claim"}],
        uncertainty=False,
        reasons=["All key claims are supported."],
    )
    judge = GroundednessJudge(llm_service=FakeJudgeLLMService(fake_output))

    result = await judge.evaluate(
        query="What is FastAPI?",
        answer="FastAPI is a Python framework.",
        context_docs=[{"source": "doc.md", "content": "FastAPI is a Python web framework."}],
    )

    assert result.name == "groundedness"
    assert result.output.score == 0.92
    assert result.output.error_type == "none"
    assert result.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_quality_judge_handles_uncertainty_flag():
    fake_output = JudgeOutput(
        score=0.4,
        decision="needs_review",
        error_type="insufficient_context",
        citations=[],
        uncertainty=True,
        reasons=["Need more context for confident scoring."],
    )
    judge = QualityJudge(llm_service=FakeJudgeLLMService(fake_output))

    result = await judge.evaluate(
        query="Explain vector databases",
        answer="They store vectors.",
        context_docs=[],
    )

    assert result.name == "quality"
    assert result.output.uncertainty is True
    assert result.output.error_type == "insufficient_context"
