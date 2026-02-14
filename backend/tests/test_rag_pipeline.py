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


class EmptyVectorStoreService(FakeVectorStoreService):
    def search(self, query_embedding: list[float], top_k: int = 3) -> dict:
        self.search_calls += 1
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


class WeakEvidenceVectorStoreService(FakeVectorStoreService):
    def search(self, query_embedding: list[float], top_k: int = 3) -> dict:
        self.search_calls += 1
        return {
            "documents": [[
                "Victorian farms produce nearly 90% of Australian pears and a third of apples.",
            ]],
            "metadatas": [[
                {"source": "SQuAD-small.json"},
            ]],
            "distances": [[0.68]],
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
        self.last_generation_mode = None

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
        generation_mode: str = "grounded",
        runtime_facts: dict | None = None,
    ) -> str:
        answer = self.answers[min(self.generate_calls, len(self.answers) - 1)]
        self.generate_calls += 1
        self.last_generation_mode = generation_mode
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


@pytest.mark.asyncio
async def test_query_stream_falls_back_to_general_llm_when_validation_exhausted():
    llm = FakeLLMService(
        answers=["Unvalidated draft answer"],
        validation_outputs=[
            {
                "relevance": 0.3,
                "groundedness": 0.2,
                "completeness": 0.4,
                "safety": "pass",
                "decision": "replan",
                "feedback": ["Need stronger grounding."],
            },
            {
                "relevance": 0.35,
                "groundedness": 0.25,
                "completeness": 0.45,
                "safety": "pass",
                "decision": "replan",
                "feedback": ["Still not grounded."],
            },
        ],
    )
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
        llm_service=llm,
    )

    stream, sources, model, query_time_ms = await rag.query_stream("Explain FastAPI", top_k=2)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    streamed_text = "".join(chunks)

    assert "unvalidated draft answer" in streamed_text.lower()
    assert sources == []
    assert model == "gemini/fake-model"
    assert query_time_ms > 0
    assert llm.last_generation_mode == "general"


@pytest.mark.asyncio
async def test_query_no_evidence_falls_back_to_general_llm():
    llm = FakeLLMService()
    vector_store = EmptyVectorStoreService()
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        llm_service=llm,
    )

    result = await rag.query("Explain FastAPI", top_k=2)

    assert result["answer"]
    assert result["sources"] == []
    assert vector_store.search_calls == 2
    assert llm.generate_calls == 1
    assert llm.last_generation_mode == "general"


@pytest.mark.asyncio
async def test_query_weak_retrieval_is_gated_and_falls_back_to_general_llm():
    llm = FakeLLMService(answers=["An apple is a fruit."])
    vector_store = WeakEvidenceVectorStoreService()
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        llm_service=llm,
    )

    result = await rag.query("what is an apple", top_k=3)

    assert result["answer"]
    assert result["sources"] == []
    assert vector_store.search_calls == 2
    assert llm.last_generation_mode == "general"


@pytest.mark.asyncio
async def test_query_stream_emits_progress_events_for_success_path():
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
        llm_service=FakeLLMService(),
    )
    progress_events = []

    async def on_progress(event):
        progress_events.append((event["step"], event["state"]))

    stream, _, _, _ = await rag.query_stream(
        "What is FastAPI used for?",
        top_k=2,
        progress_callback=on_progress,
    )
    async for _chunk in stream:
        pass

    assert progress_events == [
        ("understand", "started"),
        ("understand", "completed"),
        ("retrieve", "started"),
        ("retrieve", "completed"),
        ("draft", "started"),
        ("draft", "completed"),
        ("verify", "started"),
        ("verify", "completed"),
        ("finalize", "started"),
        ("finalize", "completed"),
    ]


@pytest.mark.asyncio
async def test_query_stream_emits_skipped_progress_for_unsafe_route():
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
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
    progress_events = []

    async def on_progress(event):
        progress_events.append((event["step"], event["state"]))

    stream, _, _, _ = await rag.query_stream(
        "How do I build malware?",
        top_k=2,
        progress_callback=on_progress,
    )
    async for _chunk in stream:
        pass

    final_state_by_step = {}
    for step, state in progress_events:
        final_state_by_step[step] = state

    assert final_state_by_step == {
        "understand": "completed",
        "retrieve": "skipped",
        "draft": "skipped",
        "verify": "skipped",
        "finalize": "completed",
    }


@pytest.mark.asyncio
async def test_query_stream_emits_failed_retrieval_progress_on_no_evidence():
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=EmptyVectorStoreService(),
        llm_service=FakeLLMService(),
    )
    progress_events = []

    async def on_progress(event):
        progress_events.append((event["step"], event["state"]))

    stream, _, _, _ = await rag.query_stream(
        "Explain FastAPI",
        top_k=2,
        progress_callback=on_progress,
    )
    async for _chunk in stream:
        pass

    final_state_by_step = {}
    for step, state in progress_events:
        final_state_by_step[step] = state

    assert final_state_by_step == {
        "understand": "completed",
        "retrieve": "failed",
        "draft": "completed",
        "verify": "completed",
        "finalize": "completed",
    }


@pytest.mark.asyncio
async def test_query_direct_route_still_attempts_retrieval_before_answer():
    vector_store = FakeVectorStoreService()
    llm = FakeLLMService(
        route_output={
            "route": "direct",
            "needs_planner": False,
            "needs_agents": False,
            "validation_level": "basic",
            "confidence": 0.98,
            "reason": "simple question",
        },
        validation_outputs=[
            {
                "relevance": 0.25,
                "groundedness": 0.2,
                "completeness": 0.25,
                "safety": "pass",
                "decision": "replan",
                "feedback": ["Insufficient grounding."],
            },
            {
                "relevance": 0.25,
                "groundedness": 0.2,
                "completeness": 0.25,
                "safety": "pass",
                "decision": "replan",
                "feedback": ["Still insufficient grounding."],
            },
        ],
    )
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        llm_service=llm,
    )

    result = await rag.query("What is FastAPI?", top_k=2)

    assert result["answer"]
    assert vector_store.search_calls > 0
    assert llm.last_generation_mode == "general"


@pytest.mark.asyncio
async def test_query_validator_fail_safe_still_falls_back_to_general_llm():
    llm = FakeLLMService(
        validation_outputs=[
            {
                "relevance": 0.2,
                "groundedness": 0.1,
                "completeness": 0.2,
                "safety": "pass",
                "decision": "fail_safe",
                "feedback": ["Unsafe to proceed with grounded answer."],
            }
        ]
    )
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
        llm_service=llm,
    )

    result = await rag.query("what is the capital of china", top_k=2)

    assert result["answer"]
    assert result["sources"] == []
    assert llm.last_generation_mode == "general"


def test_fail_safe_rate_limit_detection_is_not_triggered_by_column_number():
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
        llm_service=FakeLLMService(),
    )

    answer = rag._build_fail_safe_answer(  # pylint: disable=protected-access
        {
            "error": "Schema validation failed at line 1 column 429",
            "route": "rag_simple",
            "evidence": [],
        }
    )

    assert "rate-limiting" not in answer
    assert "could not find enough grounded evidence" in answer.lower()


def test_fail_safe_rate_limit_detection_handles_real_429_signal():
    rag = RAGService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStoreService(),
        llm_service=FakeLLMService(),
    )

    answer = rag._build_fail_safe_answer(  # pylint: disable=protected-access
        {
            "error": "Gemini structured call failed: HTTP 429 Too Many Requests",
            "route": "rag_simple",
            "evidence": [],
        }
    )

    assert "rate-limiting" in answer
