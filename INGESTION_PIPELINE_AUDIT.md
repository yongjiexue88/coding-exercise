# Ingestion Pipeline Audit Report

Date: 2026-02-14  
Workspace: `/Users/yongjiexue/Documents/GitHub/coding-exercise`

## Scope

Validate whether the current uncommitted implementation matches the stated "Advanced Ingestion Pipeline Walkthrough", and provide the practical ingest workflow plus guidance on two proposed follow-up tasks.

## Verification Summary

Overall status: **Mostly implemented, with important gaps**.

- Core architecture exists: ingest job table, blue/green run pointer, parser/chunker, worker, manager orchestration.
- Retrieval path is already integrated with the active run pointer.
- Frontend has no ingest control/polling.
- Some reliability and maintenance gaps make the implementation less "robust" than claimed.

## Plan vs Implementation Matrix

| Plan Item | Status | Evidence |
|---|---|---|
| `IngestJob` durable queue table | Implemented | `backend/models_ingest.py:19` |
| `IngestRun` versioned corpus runs | Implemented | `backend/models_ingest.py:43` |
| `CorpusState` active run pointer | Implemented | `backend/models_ingest.py:55` |
| `IngestDocument` stable source + hash | Implemented (storage only) | `backend/models_ingest.py:65`, `backend/data/pipeline/manager.py:95` |
| `IngestChunk` structure-aware chunk rows | Implemented | `backend/models_ingest.py:86`, `backend/data/pipeline/chunker.py:9` |
| `IngestChunkEmbedding` separate vectors | Implemented | `backend/models_ingest.py:101` |
| `SQuADJsonParser` title/paragraph/QA extraction | Implemented | `backend/data/pipeline/parser.py:23` |
| `SQuADChunker` Title + Content + Potential Questions | Implemented | `backend/data/pipeline/chunker.py:20` |
| Worker claim with `SKIP LOCKED` | Implemented | `backend/data/pipeline/jobs.py:118` |
| Pipeline manager + atomic switch | Implemented | `backend/data/pipeline/manager.py:67`, `backend/data/pipeline/manager.py:143` |
| Async API: `POST /ingest`, status, cancel | Implemented | `backend/main.py:155`, `backend/main.py:167`, `backend/main.py:182` |
| Retry endpoint | Implemented | `backend/main.py:191` |
| Unit tests for parser/chunker/jobs | Implemented | `backend/tests/test_ingest_components.py`, `backend/tests/test_jobs.py` |

## Gaps and Risks

### High

1. Cancellation is not durable for in-flight jobs.
- `cancel_job` can mark a processing job as cancelled (`backend/data/pipeline/jobs.py:56`).
- Worker does not stop pipeline execution on cancellation; completion can overwrite status to `completed`/`failed` (`backend/data/pipeline/jobs.py:167`).
- Practical effect: cancel is reliable for `pending`, but not truly cooperative for `processing`.

2. "No files found" can still mark job completed with no activation.
- Pipeline returns early when no JSON files exist (`backend/data/pipeline/manager.py:60`).
- No `CorpusState` switch occurs, run remains `indexing`, and worker still marks job `completed`.
- Practical effect: success signal may be misleading.

### Medium

3. Idempotency is only recorded, not enforced.
- `source_id` and `content_hash` are stored (`backend/models_ingest.py:72`, `backend/models_ingest.py:73`), but unchanged files are not skipped.
- Every run reprocesses all `*.json` files (`backend/data/pipeline/manager.py:59`).

4. Worker identity is static.
- Worker starts with hardcoded ID `api-worker-1` (`backend/main.py:53`).
- In multi-replica deployments, this weakens observability of ownership/leases.

5. Progress reporting is binary.
- API exposes `pending/processing/completed/...` but no counts/percent.
- Frontend polling can show status changes only, not progress bars.

### Maintenance/Drift

6. Legacy ingestion path conflicts with current vector store service.
- `backend/data/ingest.py` and `backend/data/ingest_squad.py` call `reset()`/`add_documents()` that current `VectorStoreService` no longer defines.
- `backend/tests/test_vector_store.py` still targets old psycopg2-style API.

7. README workflow is partly outdated.
- Local quick-start still points to `python -m data.ingest` (`README.md:165`), not the new async `/ingest` pipeline.

## Test Evidence

Executed:

```bash
PYTHONPATH=. python3 -m pytest backend/tests/test_ingest_components.py backend/tests/test_jobs.py
```

Result: **5 passed**.

Executed:

```bash
PYTHONPATH=. python3 -m pytest backend/tests/test_api.py backend/tests/test_vector_store.py
```

Result: **7 errors** due to stale mock assumptions and DB mocking mismatch with current SQLModel/engine initialization.

## Practical Data Ingest Workflow (Runbook)

### Prerequisites

1. Set backend env values:
- `GEMINI_API_KEY`
- `DATABASE_URL`

2. Put JSON source files under:
- `backend/data/documents/`

Current code ingests all `*.json` files from that folder (`backend/data/pipeline/manager.py:59`).

### Execution Steps

1. Start backend:

```bash
cd /Users/yongjiexue/Documents/GitHub/coding-exercise/backend
uvicorn main:app --reload
```

2. Trigger ingestion job:

```bash
curl -X POST http://localhost:8000/ingest
```

3. Poll job status until terminal state:

```bash
curl http://localhost:8000/ingest/<job_id>
```

Expected status lifecycle:
- `pending` -> `processing` -> `completed` (or `failed`/`cancelled`)

4. Optional cancel:

```bash
curl -X POST http://localhost:8000/ingest/<job_id>/cancel
```

5. Verify active indexed corpus:

```bash
curl http://localhost:8000/documents
curl http://localhost:8000/health
```

6. Query after ingestion:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Who created Super Mario?","top_k":3}'
```

### Internal Workflow (What Happens)

1. API creates `IngestJob`.
2. Background worker claims one job with `FOR UPDATE SKIP LOCKED`.
3. Pipeline creates a new `IngestRun` (`indexing`).
4. For each JSON file:
- parse SQuAD sections
- chunk into retrieval text with "Potential Questions"
- embed in batches
- persist `IngestDocument` + `IngestChunk` + `IngestChunkEmbedding`
5. On success, manager performs atomic switch by updating `CorpusState.active_run_id`.
6. Retrieval queries read only chunks belonging to the active run.

## Are These Two Follow-Ups Needed?

### 1) Search Integration update in `RAGService`

**Not needed as described.**  
Reason: retrieval is already active-run aware in `VectorStoreService.search()` via join through `corpus_state` (`backend/services/vector_store.py:45`-`backend/services/vector_store.py:50`). `RAGService` already calls that service (`backend/services/rag.py:298`).

When it *would* be needed:
- If you want `RAGService` to bypass `VectorStoreService` and run SQL directly (not recommended).
- If you need multi-corpus routing and explicit corpus selection per query.

### 2) Frontend "Re-Sync Data" button + polling

**Yes, recommended for usability, not required for backend correctness.**

- Backend supports async ingestion endpoints already.
- Frontend currently has no UI trigger/polling for `/ingest` (`frontend/src/App.jsx` has only query flow).
- Without it, ingestion must be triggered manually via curl/Postman.

## Recommended Next Actions

1. Add cooperative cancellation checks inside pipeline loop and in `_complete_job` to preserve `cancelled`.
2. Make no-file ingestion fail fast (`failed`) or return explicit `completed_noop`.
3. Decide idempotency policy: skip unchanged `source_id + content_hash` or keep full reindex behavior.
4. Add frontend Re-Sync control with status polling.
5. Update README to use `/ingest` async path as default.
6. Replace stale legacy tests with SQLModel-based tests for current vector store and API lifecycle.
