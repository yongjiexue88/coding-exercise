"""FastAPI application — RAG system with Gemini LLM and streaming."""

from __future__ import annotations

import json
import time
import asyncio
import logging
from uuid import UUID
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
except ImportError:  # pragma: no cover - fallback for local envs without prometheus-client
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoopMetric:
        def inc(self, *_args, **_kwargs):
            return None

        def observe(self, *_args, **_kwargs):
            return None

        def labels(self, **_kwargs):
            return self

    def Counter(*_args, **_kwargs):  # type: ignore
        return _NoopMetric()

    def Histogram(*_args, **_kwargs):  # type: ignore
        return _NoopMetric()

    def generate_latest():  # type: ignore
        return b""

from services.rag import RAGService
from data.pipeline.jobs import JobManager, JobWorker
from database import create_db_and_tables
# Import models_ingest to ensure tables are created
import models_ingest
import models_observability

from models import (
    QueryRequest,
    QueryResponse,
    IngestJobResponse,
    HealthResponse,
    DocumentInfo,
)
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG service instance
rag_service: Optional[RAGService] = None
job_worker: Optional[JobWorker] = None
worker_task: Optional[asyncio.Task] = None

QUERY_COUNT = Counter(
    "rag_query_total",
    "Total number of /query requests handled successfully.",
)
QUERY_LATENCY_MS = Histogram(
    "rag_query_latency_ms",
    "End-to-end query latency in milliseconds.",
    buckets=(50, 100, 200, 500, 1000, 2000, 3500, 5000, 10000),
)
QUERY_COST_USD = Histogram(
    "rag_query_cost_usd",
    "Estimated query cost in USD.",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.03, 0.05),
)
QUERY_PROMPT_TOKENS = Counter(
    "rag_query_prompt_tokens_total",
    "Accumulated prompt tokens across successful queries.",
)
QUERY_COMPLETION_TOKENS = Counter(
    "rag_query_completion_tokens_total",
    "Accumulated completion tokens across successful queries.",
)
QUERY_TOTAL_TOKENS = Counter(
    "rag_query_total_tokens_total",
    "Accumulated total tokens across successful queries.",
)
VALIDATOR_DECISIONS = Counter(
    "rag_validator_decision_total",
    "Validator decisions by type.",
    labelnames=("decision",),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services and worker on startup."""
    global rag_service, job_worker, worker_task

    # 1. Initialize DB Tables (including new ingestion tables)
    create_db_and_tables()

    # 2. Initialize RAG Service
    rag_service = RAGService()

    # 3. Start Background Worker
    job_worker = JobWorker(worker_id="api-worker-1")
    worker_task = asyncio.create_task(job_worker.run_loop())

    yield

    # Shutdown
    if job_worker:
        job_worker.running = False
    if worker_task:
        await worker_task


app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system with Gemini",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with model and index status."""
    doc_count = rag_service.vector_store.get_document_count() if rag_service else 0
    return HealthResponse(
        status="healthy",
        active_model=f"gemini/{settings.gemini_model}",
        documents_indexed=doc_count,
    )


# --- Query ---

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system (non-streaming)."""
    if rag_service.vector_store.get_document_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Call POST /ingest first.")

    try:
        result = await rag_service.query(request.query, top_k=request.top_k)

        observation = rag_service.last_observation or {}
        QUERY_COUNT.inc()
        QUERY_LATENCY_MS.observe(float(observation.get("query_time_ms", result.get("query_time_ms", 0.0))))
        QUERY_COST_USD.observe(float(observation.get("cost_usd", 0.0)))

        prompt_tokens = int(observation.get("prompt_tokens", 0))
        completion_tokens = int(observation.get("completion_tokens", 0))
        total_tokens = int(observation.get("total_tokens", 0))
        if prompt_tokens:
            QUERY_PROMPT_TOKENS.inc(prompt_tokens)
        if completion_tokens:
            QUERY_COMPLETION_TOKENS.inc(completion_tokens)
        if total_tokens:
            QUERY_TOTAL_TOKENS.inc(total_tokens)

        decision = str(observation.get("validator_decision", "unknown"))
        VALIDATOR_DECISIONS.labels(decision=decision).inc()

        return QueryResponse(**result)
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "API_KEY_INVALID" in error_msg:
            raise HTTPException(status_code=401, detail="Invalid or missing Gemini API key. Set GEMINI_API_KEY in .env")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {error_msg}")


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Query the RAG system with Server-Sent Events streaming."""
    if rag_service.vector_store.get_document_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Call POST /ingest first.")

    stream, sources, model_name, precomputed_ms = await rag_service.query_stream(
        request.query, top_k=request.top_k
    )

    async def event_generator():
        stream_start = time.time()
        try:
            async for chunk in stream:
                data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {data}\n\n"
        except Exception as e:
            error_data = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_data}\n\n"

        stream_ms = (time.time() - stream_start) * 1000
        elapsed_ms = round(precomputed_ms + stream_ms, 2)
        done_data = json.dumps({
            "type": "done",
            "sources": [s.model_dump() for s in sources],
            "model": model_name,
            "query_time_ms": elapsed_ms,
        })
        yield f"data: {done_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Ingestion (Async) ---

@app.post("/ingest", response_model=IngestJobResponse)
async def start_ingestion():
    """Start a background ingestion job (Async)."""
    job_id = JobManager.create_job(payload={})
    job = JobManager.get_job(job_id)
    return IngestJobResponse(
        job_id=str(job.id),
        status=job.status,
        created_at=job.created_at.isoformat()
    )


@app.get("/ingest/{job_id}", response_model=IngestJobResponse)
async def get_ingestion_status(job_id: UUID):
    """Check the status of an ingestion job."""
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return IngestJobResponse(
        job_id=str(job.id),
        status=job.status,
        created_at=job.created_at.isoformat(),
        error=job.error_details["message"] if job.error_details else None
    )


@app.post("/ingest/{job_id}/cancel")
async def cancel_ingestion(job_id: UUID):
    """Cancel a running or pending ingestion job."""
    success = JobManager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job (maybe completed/failed or not found)")
    return {"status": "cancelled"}


@app.post("/ingest/{job_id}/retry")
async def retry_ingestion(job_id: UUID):
    """Retry a failed ingestion job."""
    success = JobManager.retry_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot retry job (must be in 'failed' state)")
    return {"status": "retried", "job_id": str(job_id)}


# --- Documents ---

@app.get("/documents")
async def list_documents():
    """List all indexed document sources."""
    # This now queries the active run via VectorStoreService
    sources = rag_service.vector_store.get_all_sources()
    return [DocumentInfo(**s) for s in sources]


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
