"""RAG pipeline orchestration powered by a LangGraph state machine."""

from __future__ import annotations

import re
import time
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from sqlmodel import Session

from database import engine
from evaluation.perf_metrics import CostCalculator
from models import SourceDocument
from models_observability import QueryObservation
from services.embedding import EmbeddingService
from services.evaluation_policy import EvaluationPolicyService
from services.llm import LLMService, StructuredOutputError
from services.vector_store import VectorStoreService


class QueryState(TypedDict, total=False):
    """Shared state across graph nodes."""

    trace_id: str
    query: str
    top_k: int
    user_context: Dict[str, Any]
    route: str
    route_confidence: float
    plan: Dict[str, Any]
    plan_version: int
    tool_results: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    draft_answer: str
    final_answer: str
    validator_scores: Dict[str, float]
    validator_decision: Literal["pass", "revise", "replan", "fail_safe"]
    validator_feedback: List[str]
    retries_used: int
    token_usage: Dict[str, int]
    latency_ms: int
    stage_latency_ms: Dict[str, float]
    cost_usd: float
    eval_flags: Dict[str, Any]
    error: Optional[str]


class RAGService:
    """Orchestrates query routing, planning, tools, writing and validation."""

    MAX_RETRIES = 1
    STRUCTURED_MAX_ATTEMPTS = 2
    FALLBACK_VALIDATION_LEVEL = "strict"
    TOOL_FAILURE_RATIO_LIMIT = 0.4
    _TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
    _STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStoreService | None = None,
        llm_service: LLMService | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()
        self.llm_service = llm_service or LLMService()
        self.cost_calculator = CostCalculator()
        self.policy_service = EvaluationPolicyService()
        self.last_observation: dict[str, Any] | None = None
        self.graph = self._build_graph()

    async def query(self, query: str, top_k: int = 3) -> dict:
        """Execute the full graph and return final answer with sources."""
        start_time = time.time()
        policy = self.policy_service.get_runtime_policy()
        effective_top_k = max(top_k, policy.min_top_k)

        initial_state = self._new_state(query=query, top_k=effective_top_k)
        initial_state["eval_flags"] = policy.as_flags()

        final_state = await self.graph.ainvoke(initial_state)

        elapsed_ms = (time.time() - start_time) * 1000
        final_state["latency_ms"] = round(elapsed_ms)

        self._persist_query_observation(query=query, state=final_state, query_time_ms=elapsed_ms)

        return {
            "answer": final_state.get("final_answer", ""),
            "sources": self._build_sources_from_state(final_state),
            "model": f"gemini/{self.llm_service.current_model_name}",
            "query_time_ms": round(elapsed_ms, 2),
        }

    async def query_stream(
        self, query: str, top_k: int = 3
    ) -> tuple[AsyncIterator[str], list[SourceDocument], str]:
        """Run pre-write pipeline and return streaming writer output."""
        policy = self.policy_service.get_runtime_policy()
        state = self._new_state(query=query, top_k=max(top_k, policy.min_top_k))
        state["eval_flags"] = policy.as_flags()

        state.update(await self._ingest_query(state))
        state.update(await self._query_router(state))
        model_name = f"gemini/{self.llm_service.current_model_name}"

        if state.get("validator_decision") == "fail_safe" or state.get("route") == "unsafe":
            stream = self._single_chunk_stream(self._build_fail_safe_answer(state))
            return stream, [], model_name

        if state.get("route") not in {"direct", "clarify"}:
            state.update(await self._planner(state))
            if state.get("validator_decision") == "fail_safe":
                stream = self._single_chunk_stream(self._build_fail_safe_answer(state))
                return stream, [], model_name

            state.update(await self._tool_orchestrator(state))
            if state.get("validator_decision") == "fail_safe":
                stream = self._single_chunk_stream(self._build_fail_safe_answer(state))
                return stream, [], model_name

        context_docs = self._build_context_docs(state)
        sources = self._build_sources_from_state(state)
        stream = self.llm_service.generate_stream(
            state.get("query", query),
            context_docs,
            route=state.get("route", "rag_simple"),
        )
        return stream, sources, model_name

    def _build_graph(self):
        graph = StateGraph(QueryState)

        graph.add_node("ingest_query", self._ingest_query)
        graph.add_node("query_router", self._query_router)
        graph.add_node("planner", self._planner)
        graph.add_node("tool_orchestrator", self._tool_orchestrator)
        graph.add_node("writer", self._writer)
        graph.add_node("validation_router", self._validation_router)
        graph.add_node("decision_node", self._decision_node)
        graph.add_node("finalize_response", self._finalize_response)

        graph.add_edge(START, "ingest_query")
        graph.add_edge("ingest_query", "query_router")
        graph.add_conditional_edges(
            "query_router",
            self._next_after_router,
            {
                "planner": "planner",
                "writer": "writer",
                "finalize_response": "finalize_response",
            },
        )
        graph.add_edge("planner", "tool_orchestrator")
        graph.add_edge("tool_orchestrator", "writer")
        graph.add_edge("writer", "validation_router")
        graph.add_edge("validation_router", "decision_node")
        graph.add_conditional_edges(
            "decision_node",
            self._next_after_decision,
            {
                "planner": "planner",
                "writer": "writer",
                "finalize_response": "finalize_response",
            },
        )
        graph.add_edge("finalize_response", END)

        return graph.compile()

    async def _ingest_query(self, state: QueryState) -> Dict[str, Any]:
        started = time.time()
        result = {"query": state.get("query", "").strip()}
        return {
            **result,
            "stage_latency_ms": self._append_stage_latency(
                state, "ingest_query", (time.time() - started) * 1000
            ),
        }

    async def _query_router(self, state: QueryState) -> Dict[str, Any]:
        query = state.get("query", "")
        started = time.time()
        usage: dict[str, Any] = {}
        try:
            if hasattr(self.llm_service, "route_query_with_metadata"):
                route_result, usage = await self._call_structured_with_retry(
                    self.llm_service.route_query_with_metadata,
                    query,
                )
            else:
                route_result = await self._call_structured_with_retry(self.llm_service.route_query, query)
        except StructuredOutputError as exc:
            return {
                "route": "system_error",
                "route_confidence": 0.0,
                "validator_decision": "fail_safe",
                "validator_feedback": ["Router schema validation failed after retry."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "query_router", (time.time() - started) * 1000
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "route": "system_error",
                "route_confidence": 0.0,
                "validator_decision": "fail_safe",
                "validator_feedback": ["Router call failed."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "query_router", (time.time() - started) * 1000
                ),
            }

        route = route_result.get("route", "rag_simple")
        confidence = float(route_result.get("confidence", 0.0))
        validation_level = route_result.get("validation_level", "basic")

        if confidence < 0.6 and route != "unsafe":
            route = "rag_simple"
            validation_level = self.FALLBACK_VALIDATION_LEVEL

        eval_flags = state.get("eval_flags", {})
        if eval_flags.get("reduce_risky_routes") and route in {"direct", "tool_heavy", "rag_multi_hop"}:
            route = "rag_simple"
            validation_level = "strict"

        if eval_flags.get("force_strict_validation"):
            validation_level = "strict"

        token_usage, cost_usd = self._accumulate_usage(state, usage)

        return {
            "route": route,
            "route_confidence": confidence,
            "token_usage": token_usage,
            "cost_usd": cost_usd,
            "user_context": {
                "validation_level": validation_level,
                "router_reason": route_result.get("reason", ""),
                "needs_planner": bool(route_result.get("needs_planner", True)),
                "needs_agents": bool(route_result.get("needs_agents", False)),
            },
            "stage_latency_ms": self._append_stage_latency(
                state, "query_router", (time.time() - started) * 1000
            ),
        }

    async def _planner(self, state: QueryState) -> Dict[str, Any]:
        if state.get("validator_decision") == "fail_safe":
            return {}

        started = time.time()
        usage: dict[str, Any] = {}
        try:
            if hasattr(self.llm_service, "build_plan_with_metadata"):
                plan, usage = await self._call_structured_with_retry(
                    self.llm_service.build_plan_with_metadata,
                    query=state.get("query", ""),
                    route=state.get("route", "rag_simple"),
                    top_k=state.get("top_k", 3),
                    feedback=state.get("validator_feedback", []),
                )
            else:
                plan = await self._call_structured_with_retry(
                    self.llm_service.build_plan,
                    query=state.get("query", ""),
                    route=state.get("route", "rag_simple"),
                    top_k=state.get("top_k", 3),
                    feedback=state.get("validator_feedback", []),
                )
        except StructuredOutputError as exc:
            return {
                "validator_decision": "fail_safe",
                "validator_feedback": ["Planner schema validation failed after retry."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "planner", (time.time() - started) * 1000
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "validator_decision": "fail_safe",
                "validator_feedback": ["Planner call failed."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "planner", (time.time() - started) * 1000
                ),
            }

        if not self._is_valid_plan(plan):
            return {
                "validator_decision": "fail_safe",
                "validator_feedback": ["Planner returned malformed plan."],
                "error": "Malformed plan despite schema call.",
                "stage_latency_ms": self._append_stage_latency(
                    state, "planner", (time.time() - started) * 1000
                ),
            }

        token_usage, cost_usd = self._accumulate_usage(state, usage)
        return {
            "plan": plan,
            "plan_version": state.get("plan_version", 0) + 1,
            "token_usage": token_usage,
            "cost_usd": cost_usd,
            "stage_latency_ms": self._append_stage_latency(
                state, "planner", (time.time() - started) * 1000
            ),
        }

    async def _tool_orchestrator(self, state: QueryState) -> Dict[str, Any]:
        if state.get("validator_decision") == "fail_safe":
            return {}
        if state.get("route") in {"direct", "clarify"}:
            return {"tool_results": [], "evidence": []}

        started = time.time()

        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        budgets = plan.get("budgets", {})
        max_tool_calls = int(budgets.get("max_tool_calls", 6))
        top_k = int(state.get("top_k", 3))

        tool_calls = 0
        tool_failures = 0
        tool_results: list[dict[str, Any]] = []
        evidence = state.get("evidence", [])

        for step in steps:
            tool = step.get("tool", "")
            if tool_calls >= max_tool_calls:
                tool_results.append(
                    {
                        "step_id": step.get("id"),
                        "tool": tool,
                        "status": "skipped_budget",
                    }
                )
                continue

            try:
                if tool == "vector_search":
                    query_embedding = self.embedding_service.embed_query(
                        step.get("input", {}).get("query", state.get("query", ""))
                    )
                    search_results = self.vector_store.search(query_embedding, top_k=top_k)
                    evidence = self._extract_evidence(search_results)
                    tool_calls += 1
                    tool_results.append(
                        {
                            "step_id": step.get("id"),
                            "tool": tool,
                            "status": "ok",
                            "artifacts": {"chunks": len(evidence)},
                        }
                    )
                elif tool == "rerank":
                    evidence = self._rerank_evidence(state.get("query", ""), evidence)
                    tool_calls += 1
                    tool_results.append(
                        {
                            "step_id": step.get("id"),
                            "tool": tool,
                            "status": "ok",
                            "artifacts": {"chunks": len(evidence)},
                        }
                    )

                else:
                    tool_results.append(
                        {
                            "step_id": step.get("id"),
                            "tool": tool,
                            "status": "unknown_tool",
                        }
                    )
            except Exception as exc:
                tool_calls += 1
                tool_failures += 1
                tool_results.append(
                    {
                        "step_id": step.get("id"),
                        "tool": tool,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                if self._tool_failure_ratio(tool_failures, tool_calls) > self.TOOL_FAILURE_RATIO_LIMIT:
                    return {
                        "validator_decision": "fail_safe",
                        "validator_feedback": ["Tool orchestrator circuit breaker triggered."],
                        "tool_results": tool_results,
                        "evidence": evidence,
                        "error": "Tool circuit breaker tripped.",
                        "stage_latency_ms": self._append_stage_latency(
                            state, "tool_orchestrator", (time.time() - started) * 1000
                        ),
                    }

        if evidence and state.get("route") in {"rag_simple", "rag_multi_hop", "tool_heavy"}:
            evidence = self._rerank_evidence(state.get("query", ""), evidence)

        if not evidence:
            return {
                "validator_decision": "replan",
                "validator_feedback": ["No evidence retrieved; needs replan."],
                "tool_results": tool_results,
                "evidence": evidence,
                "stage_latency_ms": self._append_stage_latency(
                    state, "tool_orchestrator", (time.time() - started) * 1000
                ),
            }

        return {
            "tool_results": tool_results,
            "evidence": evidence,
            "stage_latency_ms": self._append_stage_latency(
                state, "tool_orchestrator", (time.time() - started) * 1000
            ),
        }

    async def _writer(self, state: QueryState) -> Dict[str, Any]:
        if state.get("validator_decision") == "fail_safe":
            return {}

        started = time.time()
        context_docs = self._build_context_docs(state)
        feedback = state.get("validator_feedback", [])

        usage: dict[str, Any] = {}
        if hasattr(self.llm_service, "generate_with_metadata"):
            draft, usage = await self.llm_service.generate_with_metadata(
                query=state.get("query", ""),
                context_docs=context_docs,
                route=state.get("route", "rag_simple"),
                feedback=feedback,
            )
        else:
            draft = await self.llm_service.generate(
                query=state.get("query", ""),
                context_docs=context_docs,
                route=state.get("route", "rag_simple"),
                feedback=feedback,
            )

        token_usage, cost_usd = self._accumulate_usage(state, usage)
        return {
            "draft_answer": draft,
            "token_usage": token_usage,
            "cost_usd": cost_usd,
            "stage_latency_ms": self._append_stage_latency(state, "writer", (time.time() - started) * 1000),
        }

    async def _validation_router(self, state: QueryState) -> Dict[str, Any]:
        if state.get("validator_decision") == "fail_safe":
            return {}

        started = time.time()
        route = state.get("route", "rag_simple")
        if route in {"direct", "clarify"}:
            return {
                "validator_scores": {"relevance": 1.0, "groundedness": 1.0, "completeness": 1.0},
                "validator_decision": "pass",
                "validator_feedback": [],
                "stage_latency_ms": self._append_stage_latency(
                    state, "validation_router", (time.time() - started) * 1000
                ),
            }

        context_docs = self._build_context_docs(state)
        usage: dict[str, Any] = {}
        try:
            if hasattr(self.llm_service, "validate_answer_with_metadata"):
                validator, usage = await self._call_structured_with_retry(
                    self.llm_service.validate_answer_with_metadata,
                    query=state.get("query", ""),
                    answer=state.get("draft_answer", ""),
                    context_docs=context_docs,
                    validation_level=state.get("user_context", {}).get("validation_level", "basic"),
                )
            else:
                validator = await self._call_structured_with_retry(
                    self.llm_service.validate_answer,
                    query=state.get("query", ""),
                    answer=state.get("draft_answer", ""),
                    context_docs=context_docs,
                    validation_level=state.get("user_context", {}).get("validation_level", "basic"),
                )
        except StructuredOutputError as exc:
            return {
                "validator_decision": "fail_safe",
                "validator_feedback": ["Validator schema validation failed after retry."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "validation_router", (time.time() - started) * 1000
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "validator_decision": "fail_safe",
                "validator_feedback": ["Validator call failed."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "validation_router", (time.time() - started) * 1000
                ),
            }

        token_usage, cost_usd = self._accumulate_usage(state, usage)
        return {
            "validator_scores": {
                "relevance": float(validator.get("relevance", 0.0)),
                "groundedness": float(validator.get("groundedness", 0.0)),
                "completeness": float(validator.get("completeness", 0.0)),
            },
            "validator_decision": validator.get("decision", "replan"),
            "validator_feedback": validator.get("feedback", []),
            "token_usage": token_usage,
            "cost_usd": cost_usd,
            "stage_latency_ms": self._append_stage_latency(
                state, "validation_router", (time.time() - started) * 1000
            ),
        }

    async def _decision_node(self, state: QueryState) -> Dict[str, Any]:
        started = time.time()
        decision = state.get("validator_decision", "replan")
        if decision in {"pass", "fail_safe"}:
            return {
                "stage_latency_ms": self._append_stage_latency(
                    state, "decision_node", (time.time() - started) * 1000
                )
            }

        retries_used = state.get("retries_used", 0)
        if retries_used >= self.MAX_RETRIES:
            return {
                "validator_decision": "fail_safe",
                "validator_feedback": [
                    "Validation failed after max retry; switching to fail-safe response."
                ],
                "stage_latency_ms": self._append_stage_latency(
                    state, "decision_node", (time.time() - started) * 1000
                ),
            }

        if decision not in {"revise", "replan"}:
            return {
                "validator_decision": "fail_safe",
                "validator_feedback": ["Unknown validator decision."],
                "stage_latency_ms": self._append_stage_latency(
                    state, "decision_node", (time.time() - started) * 1000
                ),
            }

        return {
            "retries_used": retries_used + 1,
            "stage_latency_ms": self._append_stage_latency(
                state, "decision_node", (time.time() - started) * 1000
            ),
        }

    async def _finalize_response(self, state: QueryState) -> Dict[str, Any]:
        started = time.time()
        if state.get("validator_decision") == "fail_safe":
            return {
                "final_answer": self._build_fail_safe_answer(state),
                "stage_latency_ms": self._append_stage_latency(
                    state, "finalize_response", (time.time() - started) * 1000
                ),
            }

        final_answer = state.get("draft_answer", "").strip()
        if not final_answer:
            final_answer = self._build_fail_safe_answer(state)

        return {
            "final_answer": final_answer,
            "stage_latency_ms": self._append_stage_latency(
                state, "finalize_response", (time.time() - started) * 1000
            ),
        }

    def _next_after_router(self, state: QueryState) -> Literal["planner", "writer", "finalize_response"]:
        if state.get("validator_decision") == "fail_safe" or state.get("route") == "unsafe":
            return "finalize_response"
        if state.get("route") in {"direct", "clarify"}:
            return "writer"
        return "planner"

    def _next_after_decision(self, state: QueryState) -> Literal["planner", "writer", "finalize_response"]:
        decision = state.get("validator_decision", "replan")
        if decision in {"pass", "fail_safe"}:
            return "finalize_response"
        if decision == "replan":
            return "planner"
        return "writer"

    async def _single_chunk_stream(self, text: str) -> AsyncIterator[str]:
        yield text

    async def _call_structured_with_retry(self, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for _ in range(self.STRUCTURED_MAX_ATTEMPTS):
            try:
                result = await fn(*args, **kwargs)
                return result
            except StructuredOutputError as exc:
                last_exc = exc
        if last_exc:
            raise last_exc
        raise StructuredOutputError("Unknown structured call failure.")

    def _build_fail_safe_answer(self, state: QueryState) -> str:
        error_text = (state.get("error") or "").lower()
        if "429" in error_text or "resource exhausted" in error_text:
            return (
                "The model provider is currently rate-limiting requests. "
                "Please retry in a moment."
            )

        if state.get("route") == "unsafe":
            return "I can't help with that request. Please ask a safe, informational question."

        if not state.get("evidence"):
            return (
                "I could not find enough grounded evidence in the indexed documents to answer "
                "confidently. Please rephrase the question or provide more context."
            )

        return (
            "I couldn't validate the answer with enough confidence after one retry. "
            "Please narrow the question and I can try again."
        )

    def _new_state(self, query: str, top_k: int) -> QueryState:
        return {
            "trace_id": str(uuid4()),
            "query": query,
            "top_k": top_k,
            "route": "rag_simple",
            "route_confidence": 0.0,
            "plan": {},
            "plan_version": 0,
            "tool_results": [],
            "evidence": [],
            "draft_answer": "",
            "final_answer": "",
            "validator_scores": {},
            "validator_decision": "pass",
            "validator_feedback": [],
            "retries_used": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0,
            "stage_latency_ms": {},
            "cost_usd": 0.0,
            "eval_flags": {},
            "user_context": {},
            "error": None,
        }

    def _is_valid_plan(self, plan: Any) -> bool:
        if not isinstance(plan, dict):
            return False
        steps = plan.get("steps")
        budgets = plan.get("budgets")
        if not isinstance(steps, list) or not steps:
            return False
        if not isinstance(budgets, dict):
            return False
        return True

    def _extract_evidence(self, results: dict) -> list[dict]:
        evidence: list[dict] = []
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])

        if not documents or not documents[0]:
            return evidence

        docs = documents[0]
        metas = metadatas[0] if metadatas else []
        dists = distances[0] if distances else []

        for i, doc in enumerate(docs):
            metadata = metas[i] if i < len(metas) else {}
            distance = float(dists[i]) if i < len(dists) else 1.0
            similarity = max(0.0, min(1.0, 1 - distance))
            evidence.append(
                {
                    "content": doc,
                    "source": metadata.get("source", "Unknown"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "distance": distance,
                    "relevance_score": round(similarity, 4),
                }
            )
        return evidence

    def _rerank_evidence(self, query: str, evidence: list[dict]) -> list[dict]:
        query_tokens = self._content_tokens(query)
        ranked: list[dict] = []

        for item in evidence:
            doc_tokens = self._content_tokens(item.get("content", ""))
            lexical_overlap = 0.0
            if query_tokens:
                lexical_overlap = len(query_tokens & doc_tokens) / len(query_tokens)

            semantic_score = float(item.get("relevance_score", 0.0))
            combined_score = (0.75 * semantic_score) + (0.25 * lexical_overlap)
            ranked.append({**item, "relevance_score": round(combined_score, 4)})

        ranked.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return ranked

    def _build_context_docs(self, state: QueryState) -> list[dict]:
        return [
            {"content": e.get("content", ""), "source": e.get("source", "Unknown")}
            for e in state.get("evidence", [])
        ]

    def _build_sources_from_state(self, state: QueryState) -> list[SourceDocument]:
        return [
            SourceDocument(
                content=e.get("content", ""),
                source=e.get("source", "Unknown"),
                relevance_score=float(e.get("relevance_score", 0.0)),
            )
            for e in state.get("evidence", [])
        ]

    def _tool_failure_ratio(self, failures: int, calls: int) -> float:
        if calls <= 0:
            return 0.0
        return failures / calls

    def _content_tokens(self, text: str) -> set[str]:
        tokens = {tok.lower() for tok in self._TOKEN_RE.findall(text)}
        return {tok for tok in tokens if tok not in self._STOP_WORDS}

    def _append_stage_latency(self, state: QueryState, stage: str, elapsed_ms: float) -> dict[str, float]:
        merged = dict(state.get("stage_latency_ms", {}))
        merged[stage] = round(float(merged.get(stage, 0.0)) + float(elapsed_ms), 2)
        return merged

    def _accumulate_usage(self, state: QueryState, usage: dict[str, Any]) -> tuple[dict[str, int], float]:
        current = dict(state.get("token_usage", {}))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            current[key] = int(current.get(key, 0)) + int(usage.get(key, 0) or 0)

        cost_usd = float(state.get("cost_usd", 0.0))
        if usage:
            model_name = str(usage.get("model_name") or self.llm_service.current_model_name)
            cost_usd += self.cost_calculator.estimate_cost_usd(
                model_name=model_name,
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            )

        return current, round(cost_usd, 8)

    def _persist_query_observation(self, query: str, state: QueryState, query_time_ms: float) -> None:
        payload = {
            "trace_id": state.get("trace_id", str(uuid4())),
            "query_text": query,
            "route": state.get("route", "rag_simple"),
            "validator_decision": state.get("validator_decision", "pass"),
            "query_time_ms": round(float(query_time_ms), 2),
            "total_tokens": int(state.get("token_usage", {}).get("total_tokens", 0)),
            "prompt_tokens": int(state.get("token_usage", {}).get("prompt_tokens", 0)),
            "completion_tokens": int(state.get("token_usage", {}).get("completion_tokens", 0)),
            "cost_usd": float(state.get("cost_usd", 0.0)),
            "stage_latency_json": state.get("stage_latency_ms", {}),
            "token_usage_json": state.get("token_usage", {}),
            "eval_flags_json": state.get("eval_flags", {}),
            "validator_scores_json": state.get("validator_scores", {}),
        }
        self.last_observation = payload

        try:
            with Session(engine) as session:
                session.add(QueryObservation(**payload))
                session.commit()
        except Exception:
            # Telemetry persistence must never break user-facing query execution.
            return
