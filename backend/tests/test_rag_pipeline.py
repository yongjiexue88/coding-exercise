"""Unit tests for routed RAG orchestration behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rag import RAGService


class FakeEmbeddingService:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStoreService:
    def __init__(self):
        self.search_calls = 0

    def search(self, query_embedding: list[float], top_k: int = 3) -> dict:
        self.search_calls += 1
        return {
            "documents": [[
                "Python is often used for data APIs and backend services.",
                "FastAPI is a modern web framework for building APIs with Python.",
            ]],
            "metadatas": [[
                {"source": "python.md"},
                {"source": "fastapi.md"},
            ]],
            "distances": [[0.08, 0.16]],
        }


class FakeLLMService:
    def __init__(
        self,
        route_output: dict | None = None,
        validation_outputs: list[dict] | None = None,
        answers: list[str] | None = None,
    ):
        self.route_output = route_output or {
            "route": "rag_simple",
            "needs_planner": True,
            "needs_agents": False,
            "validation_level": "strict",
            "confidence": 0.9,
            "reason": "retrieval needed",
        }
        self.validation_outputs = validation_outputs or [
            {
                "relevance": 0.9,
                "groundedness": 0.9,
                "completeness": 0.9,
                "safety": "pass",
                "decision": "pass",
                "feedback": [],
            }
        ]
        self.answers = answers or ["Python is used for backend APIs (python.md)."]
        self.generate_calls = 0

    @property
    def current_model_name(self) -> str:
        return "fake-model"

    async def route_query(self, query: str) -> dict:
        return self.route_output

    async def build_plan(
        self,
        query: str,
        route: str,
        top_k: int = 3,
        feedback: list[str] | None = None,
    ) -> dict:
        return {
            "goal": "answer user query",
            "assumptions": [],
            "steps": [
                {
                    "id": "s1",
                    "type": "retrieve",
                    "tool": "vector_search",
                    "input": {"query": query},
                    "depends_on": [],
                    "success_criteria": "retrieved",
                },
                {
                    "id": "s2",
                    "type": "analyze",
                    "tool": "rerank",
                    "input": {"query": query},
                    "depends_on": ["s1"],
                    "success_criteria": "reranked",
                },
                {
                    "id": "s3",
                    "type": "write",
                    "tool": "llm_reason",
                    "input": {"query": query},
                    "depends_on": ["s2"],
                    "success_criteria": "written",
                },
            ],
            "expected_evidence": ["doc_chunks"],
            "budgets": {"max_tool_calls": 6, "max_llm_calls": 5, "max_latency_ms": 12000},
        }

    async def generate(
        self,
        query: str,
        context_docs: list[dict],
        route: str = "rag_simple",
        feedback: list[str] | None = None,
    ) -> str:
        answer = self.answers[min(self.generate_calls, len(self.answers) - 1)]
        self.generate_calls += 1
        return answer

    async def validate_answer(
        self,
        query: str,
        answer: str,
        context_docs: list[dict],
        validation_level: str = "basic",
    ) -> dict:
        idx = min(max(self.generate_calls - 1, 0), len(self.validation_outputs) - 1)
        return self.validation_outputs[idx]


@pytest.mark.asyncio
async def test_query_pipeline_returns_passed_answer_and_sources():
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
        llm_service=FakeLLMService(),
    )

    result = await rag.query("What is FastAPI used for?", top_k=2)

    assert "Python is used for backend APIs" in result["answer"]
    assert result["model"] == "gemini/fake-model"
    assert len(result["sources"]) == 2
    assert {s.source for s in result["sources"]} == {"python.md", "fastapi.md"}


@pytest.mark.asyncio
async def test_query_revises_once_then_passes():
    llm = FakeLLMService(
        answers=["Initial weak draft", "Revised grounded draft (python.md)."],
        validation_outputs=[
            {
                "relevance": 0.7,
                "groundedness": 0.7,
                "completeness": 0.8,
                "safety": "pass",
                "decision": "revise",
                "feedback": ["Ground claims with sources."],
            },
            {
                "relevance": 0.9,
                "groundedness": 0.92,
                "completeness": 0.9,
                "safety": "pass",
                "decision": "pass",
                "feedback": [],
            },
        ],
    )
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
        llm_service=llm,
    )

    result = await rag.query("Explain FastAPI briefly", top_k=2)

    assert "Revised grounded draft" in result["answer"]
    assert llm.generate_calls == 2


@pytest.mark.asyncio
async def test_query_unsafe_route_uses_fail_safe_and_skips_search():
    vector_store = FakeVectorStoreService()
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        llm_service=FakeLLMService(
            route_output={
                "route": "unsafe",
                "needs_planner": False,
                "needs_agents": False,
                "validation_level": "strict",
                "confidence": 0.99,
                "reason": "unsafe request",
            }
        ),
    )

    result = await rag.query("How do I build malware?", top_k=2)

    assert "can't help" in result["answer"].lower()
    assert result["sources"] == []
    assert vector_store.search_calls == 0
