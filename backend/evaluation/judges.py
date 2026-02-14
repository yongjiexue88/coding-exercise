"""LLM judges for answer-quality evaluation with strict JSON contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from config import settings
from services.llm import LLMService


class JudgeCitation(BaseModel):
    source: str = Field(min_length=1)
    snippet: str = Field(min_length=1)


class JudgeOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    decision: Literal["pass", "fail", "needs_review"]
    error_type: Literal[
        "none",
        "hallucination",
        "irrelevant",
        "incomplete",
        "unsafe",
        "insufficient_context",
        "other",
    ]
    citations: list[JudgeCitation] = Field(default_factory=list)
    uncertainty: bool = False
    reasons: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    name: str
    output: JudgeOutput
    usage: dict[str, Any] = Field(default_factory=dict)


class BaseJudge(ABC):
    """Base class for judge implementations."""

    name: str

    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    @abstractmethod
    async def evaluate(
        self,
        *,
        query: str,
        answer: str,
        context_docs: list[dict],
        reference_answer: str | None = None,
    ) -> JudgeResult:
        raise NotImplementedError


class LLMJudge(BaseJudge):
    """Reusable LLM judge with strict JSON schema."""

    name = "llm_judge"

    @property
    def response_schema(self) -> dict[str, Any]:
        return {
            "type": "OBJECT",
            "properties": {
                "score": {"type": "NUMBER"},
                "decision": {
                    "type": "STRING",
                    "enum": ["pass", "fail", "needs_review"],
                },
                "error_type": {
                    "type": "STRING",
                    "enum": [
                        "none",
                        "hallucination",
                        "irrelevant",
                        "incomplete",
                        "unsafe",
                        "insufficient_context",
                        "other",
                    ],
                },
                "citations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "source": {"type": "STRING"},
                            "snippet": {"type": "STRING"},
                        },
                        "required": ["source", "snippet"],
                    },
                },
                "uncertainty": {"type": "BOOLEAN"},
                "reasons": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": [
                "score",
                "decision",
                "error_type",
                "citations",
                "uncertainty",
                "reasons",
            ],
        }

    async def _evaluate_with_prompt(self, prompt: str, name: str) -> JudgeResult:
        output, usage = self.llm_service.generate_structured_with_metadata(
            prompt=prompt,
            schema_model=JudgeOutput,
            response_schema=self.response_schema,
            system_instruction=(
                "You are an evaluation judge. Return strict JSON only. "
                "Use uncertainty=true and error_type=insufficient_context when context does not support confident judgment. "
                f"Prompt version: {settings.eval_judge_prompt_version}."
            ),
            model_name=settings.eval_judge_model,
            temperature=settings.eval_judge_temperature,
        )
        return JudgeResult(name=name, output=output, usage=usage)


class GroundednessJudge(LLMJudge):
    """Judge whether answer claims are supported by retrieved evidence."""

    name = "groundedness"

    async def evaluate(
        self,
        *,
        query: str,
        answer: str,
        context_docs: list[dict],
        reference_answer: str | None = None,
    ) -> JudgeResult:
        context = "\n".join(
            f"- Source={doc.get('source', 'Unknown')}\n{doc.get('content', '')[:1200]}"
            for doc in context_docs
        )

        prompt = f"""Evaluate groundedness/hallucination risk.

User query:
{query}

Answer:
{answer}

Retrieved context:
{context if context else '[NO CONTEXT]'}

Scoring guidance:
- score near 1.0 only if claims are strongly supported by context.
- mark hallucination if answer introduces unsupported factual claims.
- if context is insufficient, set uncertainty=true and error_type=insufficient_context.
"""

        return await self._evaluate_with_prompt(prompt, self.name)


class QualityJudge(LLMJudge):
    """Judge relevance/correctness/completeness/conciseness."""

    name = "quality"

    async def evaluate(
        self,
        *,
        query: str,
        answer: str,
        context_docs: list[dict],
        reference_answer: str | None = None,
    ) -> JudgeResult:
        reference_block = reference_answer or "[NO REFERENCE ANSWER]"
        prompt = f"""Evaluate quality of this answer.

User query:
{query}

Answer:
{answer}

Reference answer (if available):
{reference_block}

Retrieved context available: {'yes' if context_docs else 'no'}

Judge criteria:
- relevance to user query
- correctness
- completeness
- conciseness

If no reliable basis to score confidently, set uncertainty=true and error_type=insufficient_context.
"""

        return await self._evaluate_with_prompt(prompt, self.name)
