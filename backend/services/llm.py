"""Gemini LLM service with strict structured outputs for control nodes."""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel, Field, ValidationError

from config import settings


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context documents.

Rules:
1. Use only provided context for factual claims.
2. If context is insufficient, state what is missing.
3. Keep answers concise but complete.
4. Mention source document names for major claims when available.
5. Never fabricate citations or facts.
"""


class StructuredOutputError(Exception):
    """Raised when a structured Gemini response violates schema constraints."""


class RouterOutput(BaseModel):
    route: Literal["direct", "rag_simple", "rag_multi_hop", "tool_heavy", "clarify", "unsafe"]
    needs_planner: bool
    needs_agents: bool
    validation_level: Literal["basic", "strict"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)


class PlanStep(BaseModel):
    id: str = Field(min_length=1, max_length=20)
    type: Literal["retrieve", "analyze", "compute", "write"]
    tool: Literal["vector_search", "sql_query", "web_fetch", "llm_reason", "rerank", "calculator"]
    input: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: str = Field(min_length=1, max_length=300)


class PlanBudgets(BaseModel):
    max_tool_calls: int = Field(ge=1, le=20)
    max_llm_calls: int = Field(ge=1, le=20)
    max_latency_ms: int = Field(ge=1000, le=120000)


class PlannerOutput(BaseModel):
    goal: str = Field(min_length=1, max_length=500)
    assumptions: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(min_length=1)
    expected_evidence: list[str] = Field(default_factory=list)
    budgets: PlanBudgets


class ValidatorOutput(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    safety: Literal["pass", "fail", "needs_review"]
    decision: Literal["pass", "revise", "replan", "fail_safe"]
    feedback: list[str] = Field(default_factory=list)


ROUTER_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "route": {
            "type": "STRING",
            "enum": ["direct", "rag_simple", "rag_multi_hop", "tool_heavy", "clarify", "unsafe"],
        },
        "needs_planner": {"type": "BOOLEAN"},
        "needs_agents": {"type": "BOOLEAN"},
        "validation_level": {"type": "STRING", "enum": ["basic", "strict"]},
        "confidence": {"type": "NUMBER"},
        "reason": {"type": "STRING"},
    },
    "required": [
        "route",
        "needs_planner",
        "needs_agents",
        "validation_level",
        "confidence",
        "reason",
    ],
}

PLANNER_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "goal": {"type": "STRING"},
        "assumptions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "steps": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "type": {"type": "STRING", "enum": ["retrieve", "analyze", "compute", "write"]},
                    "tool": {
                        "type": "STRING",
                        "enum": [
                            "vector_search",
                            "sql_query",
                            "web_fetch",
                            "llm_reason",
                            "rerank",
                            "calculator",
                        ],
                    },
                    "input": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING"},
                            "top_k": {"type": "INTEGER"},
                            "filters": {"type": "STRING"},
                        },
                    },
                    "depends_on": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "success_criteria": {"type": "STRING"},
                },
                "required": ["id", "type", "tool", "input", "depends_on", "success_criteria"],
            },
        },
        "expected_evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
        "budgets": {
            "type": "OBJECT",
            "properties": {
                "max_tool_calls": {"type": "INTEGER"},
                "max_llm_calls": {"type": "INTEGER"},
                "max_latency_ms": {"type": "INTEGER"},
            },
            "required": ["max_tool_calls", "max_llm_calls", "max_latency_ms"],
        },
    },
    "required": ["goal", "assumptions", "steps", "expected_evidence", "budgets"],
}

VALIDATOR_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "relevance": {"type": "NUMBER"},
        "groundedness": {"type": "NUMBER"},
        "completeness": {"type": "NUMBER"},
        "safety": {"type": "STRING", "enum": ["pass", "fail", "needs_review"]},
        "decision": {"type": "STRING", "enum": ["pass", "revise", "replan", "fail_safe"]},
        "feedback": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["relevance", "groundedness", "completeness", "safety", "decision", "feedback"],
}


def _build_prompt(
    query: str,
    context_docs: list[dict],
    route: str = "rag_simple",
    feedback: list[str] | None = None,
) -> str:
    """Build answer prompt with retrieval context and optional validator feedback."""
    context_parts: list[str] = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("source", "Unknown")
        content = doc.get("content", "")
        context_parts.append(f"--- Document {i} (Source: {source}) ---\n{content}")

    context_text = "\n\n".join(context_parts) if context_parts else "[No retrieved context]"
    feedback_block = ""
    if feedback:
        feedback_block = "\nValidator feedback to address:\n" + "\n".join(f"- {item}" for item in feedback)

    return f"""Execution Route: {route}

Context Documents:
{context_text}

User Question: {query}{feedback_block}

Generate the best possible grounded answer for the user."""


class LLMService:
    """LLM service using Google Gemini."""

    def __init__(self):
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)

    @property
    def current_model_name(self) -> str:
        return settings.gemini_model

    async def generate(
        self,
        query: str,
        context_docs: list[dict],
        route: str = "rag_simple",
        feedback: list[str] | None = None,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        prompt = _build_prompt(query, context_docs, route=route, feedback=feedback)
        model = self._build_model(system_instruction=SYSTEM_PROMPT)
        response = model.generate_content(prompt)
        return getattr(response, "text", "") or ""

    async def generate_stream(
        self,
        query: str,
        context_docs: list[dict],
        route: str = "rag_simple",
        feedback: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response, yielding text chunks."""
        prompt = _build_prompt(query, context_docs, route=route, feedback=feedback)
        model = self._build_model(system_instruction=SYSTEM_PROMPT)
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def route_query(self, query: str) -> dict[str, Any]:
        """Route a query using strict schema-constrained JSON output."""
        prompt = f"""Classify this user query for a RAG system.

User Query:
{query}
"""
        output = self._generate_structured(
            prompt=prompt,
            schema_model=RouterOutput,
            response_schema=ROUTER_RESPONSE_SCHEMA,
        )

        if output.confidence < 0.6 and output.route != "unsafe":
            data = output.model_dump()
            data["route"] = "rag_simple"
            data["validation_level"] = "strict"
            data["needs_planner"] = True
            data["needs_agents"] = False
            data["reason"] = f"{output.reason} (low-confidence fallback applied)"
            return data

        return output.model_dump()

    async def build_plan(
        self,
        query: str,
        route: str,
        top_k: int = 3,
        feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a strict-schema plan for execution."""
        feedback_text = ""
        if feedback:
            feedback_text = "\nValidator feedback to address:\n" + "\n".join(f"- {item}" for item in feedback)

        prompt = f"""Create an execution plan for this route.

Route: {route}
Top K Retrieval: {top_k}
Query: {query}{feedback_text}
"""
        output = self._generate_structured(
            prompt=prompt,
            schema_model=PlannerOutput,
            response_schema=PLANNER_RESPONSE_SCHEMA,
        )
        return output.model_dump()

    async def validate_answer(
        self,
        query: str,
        answer: str,
        context_docs: list[dict],
        validation_level: str = "basic",
    ) -> dict[str, Any]:
        """Validate answer quality and return strict-schema decision."""
        context_preview = "\n".join(
            f"- {doc.get('source', 'Unknown')}: {doc.get('content', '')[:800]}"
            for doc in context_docs
        )
        prompt = f"""Validate this RAG answer.

Validation level: {validation_level}

User Query:
{query}

Answer:
{answer}

Evidence:
{context_preview if context_preview else "[No evidence]"}
"""
        output = self._generate_structured(
            prompt=prompt,
            schema_model=ValidatorOutput,
            response_schema=VALIDATOR_RESPONSE_SCHEMA,
        )
        return output.model_dump()

    def _build_model(self, system_instruction: str | None = None) -> genai.GenerativeModel:
        return genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=system_instruction,
        )

    def _generate_structured(
        self,
        prompt: str,
        schema_model: type[BaseModel],
        response_schema: dict[str, Any],
    ) -> BaseModel:
        """Generate schema-constrained JSON and validate with Pydantic."""
        model = self._build_model()
        try:
            response = model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
        except Exception as exc:
            raise StructuredOutputError(f"Gemini structured call failed: {exc}") from exc

        payload_text = (getattr(response, "text", "") or "").strip()
        if not payload_text:
            raise StructuredOutputError("Gemini returned empty structured payload.")

        try:
            return schema_model.model_validate_json(payload_text)
        except ValidationError as exc:
            raise StructuredOutputError(f"Schema validation failed: {exc}") from exc
