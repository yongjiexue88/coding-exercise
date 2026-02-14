# LangGraph + Gemini RAG Reasoning Engine Plan

## Implementation Status

- [x] LangGraph runtime integrated with compiled `StateGraph` orchestration.
- [x] Shared `QueryState` implemented and used across graph nodes.
- [x] Router implemented with strict structured output (JSON mode + schema + validation).
- [x] Planner implemented with strict structured output (JSON mode + schema + validation).
- [x] Tool orchestrator implemented (`vector_search` + `rerank`) with circuit-breaker behavior.
- [x] Writer implemented.
- [x] Validator implemented with structured output for relevance + groundedness (+ completeness/safety fields).
- [x] Decision loop implemented (`pass | revise | replan | fail_safe`) with max one retry.
- [ ] `agent_layer` for complex routes not implemented yet.
- [ ] Phase 2 and Phase 3 items not implemented yet.

## 1) Goal

Build a production-ready RAG pipeline with:

- query routing
- structured planning
- tool orchestration
- optional multi-agent collaboration
- validation gate (auditor)
- decision loop (pass, revise, replan, fail-safe)

Core principle:

- Model intelligence uses Gemini API.
- Workflow control uses LangGraph.

---

## 2) Recommended Stack

### Runtime

- Python 3.11+
- FastAPI (serving)
- LangGraph (state machine + conditional edges)
- LangChain core utilities (optional, for tool wrappers/messages)
- Gemini API SDK (`google-genai`) or LangChain Gemini adapter (`langchain-google-genai`)

### Data + Infra

- Single database: Neon PostgreSQL
- Vector store inside Neon: `pgvector` extension (`vector` type + HNSW/IVFFlat index)
- Redis (optional) for cache/session/rate limiting
- Relational tables in same Neon database for app state/logs/evals
- Docker + Cloud Run (or equivalent) for deployment

### Observability

- LangSmith or OpenTelemetry traces
- Structured JSON logs
- Metrics: latency, token usage, retrieval quality, validation pass rate

---

## 3) Target Execution Flow

`query -> router -> planner -> tools/agents -> draft -> validator -> (pass | revise | replan | fail-safe)`

LangGraph nodes:

1. `ingest_query`
2. `query_router`
3. `planner`
4. `tool_orchestrator`
5. `agent_layer` (conditional, complex queries only)
6. `writer`
7. `validation_router`
8. `decision_node`
9. `finalize_response`

---

## 4) Query Router Design

There is no separate router API. Use Gemini with strict JSON schema output.

### Router output schema

```json
{
  "route": "direct|rag_simple|rag_multi_hop|tool_heavy|clarify|unsafe",
  "needs_planner": true,
  "needs_agents": false,
  "validation_level": "basic|strict",
  "confidence": 0.0,
  "reason": "short explanation"
}
```

### Router rules

- `confidence < 0.6` => force `rag_simple` + `strict` validation.
- `unsafe` => safe completion path.
- `direct` => skip heavy planner/tools unless user explicitly asks for deep analysis.

---

## 5) Planner (Reasoning Engine) Design

Planner also uses Gemini structured output with hard schema validation.

### Plan schema

```json
{
  "goal": "string",
  "assumptions": ["string"],
  "steps": [
    {
      "id": "s1",
      "type": "retrieve|analyze|compute|write",
      "tool": "vector_search|sql_query|web_fetch|llm_reason",
      "input": {"query": "string"},
      "depends_on": [],
      "success_criteria": "string"
    }
  ],
  "expected_evidence": ["citations", "table", "doc_chunks"],
  "budgets": {
    "max_tool_calls": 6,
    "max_llm_calls": 5,
    "max_latency_ms": 12000
  }
}
```

### Planner safeguards

- Reject malformed plans (schema fail => one retry, then fail-safe).
- Enforce budget limits before each node execution.
- Persist plan version for audit (`plan_v1`, `plan_v2`).

---

## 6) Tool Orchestrator

Implement as LangGraph node that:

- executes each planned step in topological order
- supports retries with exponential backoff
- applies per-tool timeout and global request budget
- records artifacts (retrieved chunks, tool outputs, errors)

### Minimum tool set

1. `vector_search(query, top_k)`
2. `sql_lookup(query_or_filters)` for relational metadata/business data in Neon
3. `rerank(chunks, query)`
4. `calculator` (if numerical tasks exist)

### Controls

- Retry policy: max 2 retries per tool
- Timeout per tool: 1-3s default
- Circuit breaker: abort tool chain if >40% tool failures

---

## 7) Agent Layer (Complex Queries Only)

Only run when route is `rag_multi_hop` or `tool_heavy` and budget allows.

### Agents

1. Retriever Agent
- rewrites query
- selects sources/indexes
- requests missing evidence

2. Analyst Agent
- synthesizes evidence
- detects contradictions
- marks uncertain claims

3. Writer Agent
- drafts final response
- includes citation mapping for each major claim

### Coordination

- Shared state object, no free-form infinite loops
- Max 1 collaboration round initially

---

## 8) Validation Router (Gatekeeper/Auditor)

Validation node outputs a scorecard and decision.

### Required checks (phase 1)

1. Relevance score (query vs answer)
2. Groundedness score (claims backed by evidence)

### Later checks (phase 2+)

3. Completeness (all user asks covered)
4. Safety/policy
5. Citation integrity (citation points to actual evidence)

### Validation output schema

```json
{
  "relevance": 0.0,
  "groundedness": 0.0,
  "completeness": 0.0,
  "safety": "pass|fail|needs_review",
  "decision": "pass|revise|replan|fail_safe",
  "feedback": ["string"]
}
```

### Decision thresholds (initial)

- `pass` if relevance >= 0.8 and groundedness >= 0.8
- `revise` if 0.6-0.79 on either
- `replan` if < 0.6 or missing required evidence
- `fail_safe` on unsafe/policy fail

---

## 9) Shared QueryState (LangGraph State)

Use one typed state object across all nodes.

```python
from typing import Any, Dict, List, Literal, Optional, TypedDict

class QueryState(TypedDict, total=False):
    query: str
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
    error: Optional[str]
```

---

## 10) Practical Implementation Order

Start simple. Do not enable full multi-agent on day 1.

### Phase 1 (MVP, 1-2 weeks)

- [x] 1. Build `QueryState`.
- [x] 2. Implement `query_router` with structured output.
- [x] 3. Implement `planner` with strict schema.
- [x] 4. Implement basic `tool_orchestrator` (retrieve + rerank).
- [x] 5. Implement `writer`.
- [x] 6. Implement two validators: relevance + groundedness.
- [x] 7. Add single retry loop:
  - [x] validator fail -> planner feedback -> regenerate once

### Phase 2 (Hard-query support)

1. Add conditional `agent_layer` for complex routes only.
2. Add completeness validator.
3. Add citation integrity check.
4. Add richer failure policies (replan limits, fallback responses).

### Phase 3 (Production hardening)

1. Add observability dashboards + tracing.
2. Add eval dataset and regression tests.
3. Add caching and rate limiting.
4. Tune latency/cost with dynamic model selection.

---

## 11) Model Strategy (Gemini)

Suggested starting policy:

- Router + validators: fast Gemini model
- Planner + writer: higher quality Gemini model for complex queries
- Simple route can stay fully on fast model

All model calls remain Gemini API calls; LangGraph only orchestrates.

---

## 12) Latency Budget (Realistic)

Target interactive SLA: 6-12s for complex queries, 2-5s for simple queries.

Typical breakdown:

- Router: 0.2-0.8s
- Planner: 0.6-2.0s
- Retrieval/rerank: 0.2-1.0s
- Writer: 0.8-3.0s
- Validation: 0.4-1.5s

With one revise/replan loop, total can reach 10-25s. Enforce global timeout.

---

## 13) Deployment Blueprint

### Service topology

1. `api-service` (FastAPI + LangGraph runtime)
2. `neon-postgres` (single DB: relational + pgvector)
3. `redis` (optional cache/queues)

### Deployment steps

1. Containerize `api-service`.
2. Inject Gemini API key via secret manager.
3. Set request timeout (for example 20s hard cap).
4. Configure autoscaling and concurrency limits.
5. Add health checks and structured logging.

---

## 14) Acceptance Criteria

MVP is ready when:

- Router returns valid schema >= 99% of requests.
- Planner schema validation pass >= 98%.
- Validation gate catches low-grounded answers before response.
- End-to-end success rate meets target on your eval set.
- P95 latency is within agreed SLA for each route type.

---

## 15) First Build Scope (Recommended)

Implement this exact subset first:

1. One Gemini model family
2. Router + Planner + Tool Orchestrator + Writer
3. Relevance + Groundedness validator
4. Single retry loop (max 1)
5. No multi-agent unless route is complex

This gives you controllability and auditability without overbuilding.
