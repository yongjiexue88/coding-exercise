"""Tests that query observations are captured and persisted."""

import pytest

from services.rag import RAGService


class FakeEmbeddingService:
    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3]


class FakeVectorStoreService:
    def search(self, query_embedding, top_k=3):
        return {
            "documents": [["Document content"]],
            "metadatas": [[{"source": "doc.md", "chunk_id": "abc", "chunk_index": 0}]],
            "distances": [[0.1]],
        }


class FakeLLMService:
    @property
    def current_model_name(self):
        return "fake-model"

    async def route_query_with_metadata(self, query: str):
        return (
            {
                "route": "rag_simple",
                "needs_planner": True,
                "needs_agents": False,
                "validation_level": "strict",
                "confidence": 0.95,
                "reason": "retrieval needed",
            },
            {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "model_name": "fake-model",
            },
        )

    async def build_plan_with_metadata(self, query: str, route: str, top_k=3, feedback=None):
        return (
            {
                "goal": "answer query",
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
                ],
                "expected_evidence": ["chunks"],
                "budgets": {"max_tool_calls": 4, "max_llm_calls": 3, "max_latency_ms": 12000},
            },
            {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
                "model_name": "fake-model",
            },
        )

    async def generate_with_metadata(
        self,
        query: str,
        context_docs,
        route="rag_simple",
        feedback=None,
        generation_mode="grounded",
        runtime_facts=None,
    ):
        return (
            "Grounded answer (doc.md).",
            {
                "prompt_tokens": 12,
                "completion_tokens": 6,
                "total_tokens": 18,
                "model_name": "fake-model",
            },
        )

    async def validate_answer_with_metadata(self, query: str, answer: str, context_docs, validation_level="strict"):
        return (
            {
                "relevance": 0.9,
                "groundedness": 0.92,
                "completeness": 0.88,
                "safety": "pass",
                "decision": "pass",
                "feedback": [],
            },
            {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "model_name": "fake-model",
            },
        )


@pytest.mark.asyncio
async def test_query_observation_persisted(monkeypatch):
    persisted = []

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def exec(self, stmt):
            persisted.append(stmt)
            return None

        def commit(self):
            return None

    monkeypatch.setattr("services.rag.Session", FakeSession)

    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
        llm_service=FakeLLMService(),
    )

    result = await rag.query("What is this?", top_k=2)

    assert "answer" in result
    assert rag.last_observation is not None
    assert rag.last_observation["route"] == "rag_simple"
    assert rag.last_observation["total_tokens"] > 0
    assert persisted, "Expected QueryObservation to be persisted"
