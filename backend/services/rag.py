"""RAG pipeline orchestration powered by a LangGraph state machine."""

from __future__ import annotations

import inspect
import logging
import re
import time
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Literal, Optional, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from sqlalchemy import insert
from sqlmodel import Session

from database import engine
from evaluation.perf_metrics import CostCalculator
from models import SourceDocument
from models_observability import QueryObservation
from services.embedding import EmbeddingService
from services.evaluation_policy import EvaluationPolicyService
from services.llm import LLMService, StructuredOutputError
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

ProgressStep = Literal["understand", "retrieve", "draft", "verify", "finalize"]
ProgressStatus = Literal["started", "completed", "failed", "skipped"]
ProgressCallback = Callable[[dict[str, Any]], Any]


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
    _progress_timeline: Dict[str, str]
    _progress_callback: Optional[ProgressCallback]


class RAGService:
    """Orchestrates query routing, planning, tools, writing and validation."""

    MAX_RETRIES = 1
    STRUCTURED_MAX_ATTEMPTS = 2
    FALLBACK_VALIDATION_LEVEL = "strict"
    TOOL_FAILURE_RATIO_LIMIT = 0.4
    PROGRESS_LABELS: dict[ProgressStep, str] = {
        "understand": "Understanding question",
        "retrieve": "Finding relevant sources",
        "draft": "Drafting answer",
        "verify": "Verifying answer",
        "finalize": "Finalizing response",
    }
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
        final_state, elapsed_ms = await self._run_graph(query=query, top_k=top_k)

        self._persist_query_observation(query=query, state=final_state, query_time_ms=elapsed_ms)

        return {
            "answer": final_state.get("final_answer", ""),
            "sources": self._build_sources_from_state(final_state),
            "model": f"gemini/{self.llm_service.current_model_name}",
            "query_time_ms": round(elapsed_ms, 2),
        }

    async def query_stream(
        self,
        query: str,
        top_k: int = 3,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[AsyncIterator[str], list[SourceDocument], str, float]:
        """Run the same validated graph as /query and stream the final answer text."""
        final_state, elapsed_ms = await self._run_graph(
            query=query,
            top_k=top_k,
            progress_callback=progress_callback,
        )
        self._persist_query_observation(query=query, state=final_state, query_time_ms=elapsed_ms)

        model_name = f"gemini/{self.llm_service.current_model_name}"
        sources = self._build_sources_from_state(final_state)
        stream = self._chunked_text_stream(final_state.get("final_answer", ""))
        return stream, sources, model_name, round(elapsed_ms, 2)

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
        graph.add_conditional_edges(
            "tool_orchestrator",
            self._next_after_tools,
            {
                "planner": "planner",
                "writer": "writer",
                "finalize_response": "finalize_response",
            },
        )
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
        updates: dict[str, Any] = {}
        await self._add_progress_update(state, updates, "understand", "started")
        result = {"query": state.get("query", "").strip()}
        return {
            **updates,
            **result,
            "stage_latency_ms": self._append_stage_latency(
                state, "ingest_query", (time.time() - started) * 1000
            ),
        }

    async def _query_router(self, state: QueryState) -> Dict[str, Any]:
        query = state.get("query", "")
        started = time.time()
        usage: dict[str, Any] = {}
        updates: dict[str, Any] = {}
        try:
            if hasattr(self.llm_service, "route_query_with_metadata"):
                route_result, usage = await self._call_structured_with_retry(
                    self.llm_service.route_query_with_metadata,
                    query,
                )
            else:
                route_result = await self._call_structured_with_retry(self.llm_service.route_query, query)
        except StructuredOutputError as exc:
            await self._add_progress_update(state, updates, "understand", "failed")
            await self._add_progress_update(state, updates, "retrieve", "skipped")
            await self._add_progress_update(state, updates, "draft", "skipped")
            await self._add_progress_update(state, updates, "verify", "skipped")
            return {
                **updates,
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
            await self._add_progress_update(state, updates, "understand", "failed")
            await self._add_progress_update(state, updates, "retrieve", "skipped")
            await self._add_progress_update(state, updates, "draft", "skipped")
            await self._add_progress_update(state, updates, "verify", "skipped")
            return {
                **updates,
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
        if eval_flags.get("reduce_risky_routes") and route in {"direct"}:
            route = "rag_simple"
            validation_level = "strict"

        if eval_flags.get("force_strict_validation"):
            validation_level = "strict"

        token_usage, cost_usd = self._accumulate_usage(state, usage)
        await self._add_progress_update(state, updates, "understand", "completed")
        if route == "unsafe":
            await self._add_progress_update(
                state, updates, "retrieve", "skipped", meta={"reason": "safety_policy"}
            )
            await self._add_progress_update(
                state, updates, "draft", "skipped", meta={"reason": "safety_policy"}
            )
            await self._add_progress_update(
                state, updates, "verify", "skipped", meta={"reason": "safety_policy"}
            )
        elif route in {"direct", "clarify"}:
            await self._add_progress_update(
                state,
                updates,
                "retrieve",
                "skipped",
                meta={"reason": "route_bypassed_retrieval"},
            )

        return {
            **updates,
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
        updates: dict[str, Any] = {}
        await self._add_progress_update(state, updates, "retrieve", "started")
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
            await self._add_progress_update(state, updates, "retrieve", "failed")
            await self._add_progress_update(state, updates, "draft", "skipped")
            await self._add_progress_update(state, updates, "verify", "skipped")
            return {
                **updates,
                "validator_decision": "fail_safe",
                "validator_feedback": ["Planner schema validation failed after retry."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "planner", (time.time() - started) * 1000
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            await self._add_progress_update(state, updates, "retrieve", "failed")
            await self._add_progress_update(state, updates, "draft", "skipped")
            await self._add_progress_update(state, updates, "verify", "skipped")
            return {
                **updates,
                "validator_decision": "fail_safe",
                "validator_feedback": ["Planner call failed."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "planner", (time.time() - started) * 1000
                ),
            }

        if not self._is_valid_plan(plan):
            await self._add_progress_update(state, updates, "retrieve", "failed")
            await self._add_progress_update(state, updates, "draft", "skipped")
            await self._add_progress_update(state, updates, "verify", "skipped")
            return {
                **updates,
                "validator_decision": "fail_safe",
                "validator_feedback": ["Planner returned malformed plan."],
                "error": "Malformed plan despite schema call.",
                "stage_latency_ms": self._append_stage_latency(
                    state, "planner", (time.time() - started) * 1000
                ),
            }

        token_usage, cost_usd = self._accumulate_usage(state, usage)
        return {
            **updates,
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
            updates: dict[str, Any] = {}
            await self._add_progress_update(state, updates, "retrieve", "skipped")
            return {**updates, "tool_results": [], "evidence": []}

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
        updates: dict[str, Any] = {}

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
                    await self._add_progress_update(state, updates, "retrieve", "failed")
                    await self._add_progress_update(state, updates, "draft", "skipped")
                    await self._add_progress_update(state, updates, "verify", "skipped")
                    return {
                        **updates,
                        "validator_decision": "fail_safe",
                        "validator_feedback": ["Tool orchestrator circuit breaker triggered."],
                        "tool_results": tool_results,
                        "evidence": evidence,
                        "error": "Tool circuit breaker tripped.",
                        "stage_latency_ms": self._append_stage_latency(
                            state, "tool_orchestrator", (time.time() - started) * 1000
                        ),
                    }

        if evidence and state.get("route") in {"rag_simple"}:
            evidence = self._rerank_evidence(state.get("query", ""), evidence)

        if not evidence:
            retries_used = int(state.get("retries_used", 0))
            if retries_used >= self.MAX_RETRIES:
                await self._add_progress_update(state, updates, "retrieve", "failed")
                await self._add_progress_update(state, updates, "draft", "skipped")
                await self._add_progress_update(state, updates, "verify", "skipped")
                return {
                    **updates,
                    "validator_decision": "fail_safe",
                    "validator_feedback": [
                        "Retrieval failed after max retry; switching to fail-safe response."
                    ],
                    "tool_results": tool_results,
                    "evidence": evidence,
                    "stage_latency_ms": self._append_stage_latency(
                        state, "tool_orchestrator", (time.time() - started) * 1000
                    ),
                }
            return {
                **updates,
                "validator_decision": "replan",
                "validator_feedback": ["No evidence retrieved; needs replan."],
                "retries_used": retries_used + 1,
                "tool_results": tool_results,
                "evidence": evidence,
                "stage_latency_ms": self._append_stage_latency(
                    state, "tool_orchestrator", (time.time() - started) * 1000
                ),
            }

        await self._add_progress_update(
            state,
            updates,
            "retrieve",
            "completed",
            meta={"sources": len(evidence)},
        )
        return {
            **updates,
            "validator_decision": "pass",
            "tool_results": tool_results,
            "evidence": evidence,
            "stage_latency_ms": self._append_stage_latency(
                state, "tool_orchestrator", (time.time() - started) * 1000
            ),
        }

    async def _writer(self, state: QueryState) -> Dict[str, Any]:
        if state.get("validator_decision") in {"fail_safe", "replan"}:
            return {}

        started = time.time()
        updates: dict[str, Any] = {}
        await self._add_progress_update(state, updates, "draft", "started")
        context_docs = self._build_context_docs(state)
        feedback = state.get("validator_feedback", [])

        usage: dict[str, Any] = {}
        try:
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
        except Exception:
            await self._add_progress_update(state, updates, "draft", "failed")
            raise

        token_usage, cost_usd = self._accumulate_usage(state, usage)
        await self._add_progress_update(state, updates, "draft", "completed")
        return {
            **updates,
            "draft_answer": draft,
            "token_usage": token_usage,
            "cost_usd": cost_usd,
            "stage_latency_ms": self._append_stage_latency(state, "writer", (time.time() - started) * 1000),
        }

    async def _validation_router(self, state: QueryState) -> Dict[str, Any]:
        if state.get("validator_decision") == "fail_safe":
            return {}

        started = time.time()
        updates: dict[str, Any] = {}
        await self._add_progress_update(state, updates, "verify", "started")
        route = state.get("route", "rag_simple")
        if route in {"direct", "clarify"}:
            return {
                **updates,
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
            await self._add_progress_update(state, updates, "verify", "failed")
            return {
                **updates,
                "validator_decision": "fail_safe",
                "validator_feedback": ["Validator schema validation failed after retry."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "validation_router", (time.time() - started) * 1000
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            await self._add_progress_update(state, updates, "verify", "failed")
            return {
                **updates,
                "validator_decision": "fail_safe",
                "validator_feedback": ["Validator call failed."],
                "error": str(exc),
                "stage_latency_ms": self._append_stage_latency(
                    state, "validation_router", (time.time() - started) * 1000
                ),
            }

        token_usage, cost_usd = self._accumulate_usage(state, usage)
        return {
            **updates,
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
        updates: dict[str, Any] = {}
        decision = state.get("validator_decision", "replan")
        if decision in {"pass", "fail_safe"}:
            if decision == "pass":
                await self._add_progress_update(state, updates, "verify", "completed")
            else:
                await self._add_progress_update(state, updates, "verify", "failed")
            return {
                **updates,
                "stage_latency_ms": self._append_stage_latency(
                    state, "decision_node", (time.time() - started) * 1000
                )
            }

        retries_used = state.get("retries_used", 0)
        if retries_used >= self.MAX_RETRIES:
            await self._add_progress_update(state, updates, "verify", "failed")
            return {
                **updates,
                "validator_decision": "fail_safe",
                "validator_feedback": [
                    "Validation failed after max retry; switching to fail-safe response."
                ],
                "stage_latency_ms": self._append_stage_latency(
                    state, "decision_node", (time.time() - started) * 1000
                ),
            }

        if decision not in {"revise", "replan"}:
            await self._add_progress_update(state, updates, "verify", "failed")
            return {
                **updates,
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
        updates: dict[str, Any] = {}
        await self._add_progress_update(state, updates, "finalize", "started")
        merged_state: QueryState = {**state, **updates}
        unresolved = self._resolve_progress_final_states(merged_state)
        for step, status in unresolved.items():
            await self._add_progress_update(state, updates, step, status)

        if state.get("validator_decision") == "fail_safe":
            await self._add_progress_update(state, updates, "finalize", "completed")
            return {
                **updates,
                "final_answer": self._build_fail_safe_answer(state),
                "stage_latency_ms": self._append_stage_latency(
                    state, "finalize_response", (time.time() - started) * 1000
                ),
            }

        final_answer = state.get("draft_answer", "").strip()
        if not final_answer:
            final_answer = self._build_fail_safe_answer(state)
        await self._add_progress_update(state, updates, "finalize", "completed")

        return {
            **updates,
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

    def _next_after_tools(self, state: QueryState) -> Literal["planner", "writer", "finalize_response"]:
        decision = state.get("validator_decision", "pass")
        if decision == "fail_safe":
            return "finalize_response"
        if decision == "replan":
            return "planner"
        return "writer"

    def _next_after_decision(self, state: QueryState) -> Literal["planner", "writer", "finalize_response"]:
        decision = state.get("validator_decision", "replan")
        if decision in {"pass", "fail_safe"}:
            return "finalize_response"
        if decision == "replan":
            return "planner"
        return "writer"

    async def _single_chunk_stream(self, text: str) -> AsyncIterator[str]:
        yield text

    async def _chunked_text_stream(self, text: str, chunk_size: int = 96) -> AsyncIterator[str]:
        if not text:
            return
        for idx in range(0, len(text), chunk_size):
            yield text[idx : idx + chunk_size]

    async def _run_graph(
        self,
        query: str,
        top_k: int,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[QueryState, float]:
        start_time = time.time()
        policy = self.policy_service.get_runtime_policy()
        effective_top_k = max(top_k, policy.min_top_k)

        initial_state = self._new_state(query=query, top_k=effective_top_k)
        initial_state["eval_flags"] = policy.as_flags()
        if progress_callback:
            initial_state["_progress_callback"] = progress_callback
        final_state = await self.graph.ainvoke(initial_state)
        final_state.pop("_progress_callback", None)

        elapsed_ms = (time.time() - start_time) * 1000
        final_state["latency_ms"] = round(elapsed_ms)
        return final_state, elapsed_ms

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

        if self._is_rate_limited_error(error_text):
            logger.warning("Fail-safe classified as rate limit: %s", error_text[:400])
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

    def _is_rate_limited_error(self, error_text: str) -> bool:
        if not error_text:
            return False
        if "resource exhausted" in error_text:
            return True
        if "too many requests" in error_text:
            return True
        if "rate limit" in error_text:
            return True
        if "quota exceeded" in error_text:
            return True
        return bool(re.search(r"(?:status(?:\\s*code)?|http)\\s*[:=]?\\s*429\\b", error_text))

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
            "_progress_timeline": {},
        }

    async def _add_progress_update(
        self,
        state: QueryState,
        updates: dict[str, Any],
        step: ProgressStep,
        status: ProgressStatus,
        meta: dict[str, Any] | None = None,
    ) -> None:
        merged_state: QueryState = {**state, **updates}
        progress_update = await self._set_progress_state(merged_state, step, status, meta=meta)
        updates.update(progress_update)

    async def _set_progress_state(
        self,
        state: QueryState,
        step: ProgressStep,
        status: ProgressStatus,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timeline = dict(state.get("_progress_timeline", {}))
        current = timeline.get(step)
        terminal_states = {"completed", "failed", "skipped"}

        if current == status:
            return {"_progress_timeline": timeline}
        if current in terminal_states and status == "started":
            return {"_progress_timeline": timeline}
        if current in terminal_states and status in terminal_states:
            return {"_progress_timeline": timeline}

        timeline[step] = status
        callback = state.get("_progress_callback")
        if callback:
            event: dict[str, Any] = {
                "step": step,
                "state": status,
                "label": self.PROGRESS_LABELS[step],
            }
            if meta:
                event["meta"] = meta

            callback_result = callback(event)
            if inspect.isawaitable(callback_result):
                await callback_result

        return {"_progress_timeline": timeline}

    def _resolve_progress_final_states(self, state: QueryState) -> dict[ProgressStep, ProgressStatus]:
        timeline = dict(state.get("_progress_timeline", {}))
        route = state.get("route", "rag_simple")
        decision = state.get("validator_decision", "pass")
        evidence = state.get("evidence", [])
        has_evidence = bool(evidence)
        has_draft = bool((state.get("draft_answer") or "").strip())

        resolved: dict[ProgressStep, ProgressStatus] = {}

        def _resolve(step: ProgressStep, terminal_status: ProgressStatus) -> None:
            current = timeline.get(step)
            if current in {"completed", "failed", "skipped"}:
                return
            if current != terminal_status:
                resolved[step] = terminal_status

        _resolve("understand", "completed")

        if route in {"unsafe", "direct", "clarify"}:
            _resolve("retrieve", "skipped")
        elif has_evidence:
            _resolve("retrieve", "completed")
        elif decision == "fail_safe":
            _resolve("retrieve", "failed")
        else:
            _resolve("retrieve", "skipped")

        if route == "unsafe":
            _resolve("draft", "skipped")
        elif has_draft:
            _resolve("draft", "completed")
        else:
            _resolve("draft", "skipped")

        if route == "unsafe" or not has_draft:
            _resolve("verify", "skipped")
        elif decision == "fail_safe":
            _resolve("verify", "failed")
        elif decision == "pass":
            _resolve("verify", "completed")
        else:
            _resolve("verify", "skipped")

        return resolved

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
            "created_at": datetime.utcnow(),
        }
        self.last_observation = payload

        try:
            with Session(engine) as session:
                session.exec(insert(QueryObservation).values(**payload))
                session.commit()
        except Exception:
            # Telemetry persistence must never break user-facing query execution.
            return
