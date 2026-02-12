"""FastAPI application — RAG system with Gemini LLM and streaming."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from services.rag import RAGService
from data.ingest import ingest
from models import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    HealthResponse,
    DocumentInfo,
)
from config import settings


# Global RAG service instance
rag_service: Optional[RAGService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global rag_service
    rag_service = RAGService()
    
    # Check if vector store is empty (typical for fresh Cloud Run instance)
    if rag_service.vector_store.get_document_count() == 0:
        print("🚀 Startup: Vector store is empty. Ingesting documents...")
        ingest(reset=True)
        # Re-initialize vector store to catch the new data
        rag_service.vector_store = rag_service.vector_store.__class__()
        print(f"✅ Startup: Ingestion complete. {rag_service.vector_store.get_document_count()} documents indexed.")
    
    yield


app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system with Gemini",
    version="1.0.0",
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
    doc_count = rag_service.vector_store.get_document_count()
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

    stream, sources, model_name = await rag_service.query_stream(
        request.query, top_k=request.top_k
    )

    async def event_generator():
        start_time = time.time()
        try:
            async for chunk in stream:
                data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {data}\n\n"
        except Exception as e:
            error_data = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_data}\n\n"

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
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


# --- Documents ---

@app.get("/documents")
async def list_documents():
    """List all indexed document sources."""
    sources = rag_service.vector_store.get_all_sources()
    return [DocumentInfo(**s) for s in sources]


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """Ingest documents from the data/documents directory."""
    try:
        stats = ingest(reset=True)
        # Reinitialize vector store after ingestion
        rag_service.vector_store = rag_service.vector_store.__class__()
        return IngestResponse(
            status="success",
            documents_processed=stats["documents_processed"],
            chunks_created=stats["chunks_created"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
