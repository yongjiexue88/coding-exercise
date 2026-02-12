"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request body for RAG query endpoints."""
    query: str = Field(..., min_length=1, max_length=1000, description="The user's question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")


class SourceDocument(BaseModel):
    """A retrieved source document with relevance score."""
    content: str
    source: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response from the RAG query endpoint."""
    answer: str
    sources: list[SourceDocument]
    model: str
    query_time_ms: float


class IngestResponse(BaseModel):
    """Response from the document ingestion endpoint."""
    status: str
    documents_processed: int
    chunks_created: int


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    source: str
    chunk_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    active_model: str
    documents_indexed: int
