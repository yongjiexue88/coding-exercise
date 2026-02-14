"""SQLModel definitions for the advanced data ingestion pipeline.

Includes:
- Durable Job Queue (IngestJob)
- Blue/Green Deployment (IngestRun, CorpusState)
- Structure-Aware Data (IngestDocument, IngestChunk, IngestChunkEmbedding)
"""

from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field, Relationship, JSON, Column
from pgvector.sqlalchemy import Vector
from sqlalchemy import UniqueConstraint, Index


# --- Job Management ---

class IngestJob(SQLModel, table=True):
    """A durable background job for ingestion."""
    __tablename__ = "ingest_job"
    
    id: Optional[UUID] = Field(default=None, primary_key=True)
    status: str = Field(default="pending", index=True)  # pending, processing, completed, failed, cancelled
    worker_id: Optional[str] = Field(default=None)
    lease_expires_at: Optional[datetime] = Field(default=None, index=True)
    
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
    
    payload: Optional[dict] = Field(default=None, sa_type=JSON)  # Arguments for the job
    error_details: Optional[dict] = Field(default=None, sa_type=JSON)
    retry_count: int = Field(default=0)

    # Index for checking stale leases: (status, lease_expires_at)
    __table_args__ = (
        Index("idx_ingest_job_lease", "status", "lease_expires_at"),
    )


# --- Blue/Green Deployment ---

class IngestRun(SQLModel, table=True):
    """Represents a full ingestion run (a specific version of the corpus)."""
    __tablename__ = "ingest_run"

    id: Optional[UUID] = Field(default=None, primary_key=True)
    corpus_name: str = Field(default="default", index=True)
    status: str = Field(default="indexing")  # indexing, ready, archived
    created_at: Optional[datetime] = Field(default=None)
    
    documents: List["IngestDocument"] = Relationship(back_populates="run")


class CorpusState(SQLModel, table=True):
    """Pointer to the currently active IngestRun for a given corpus."""
    __tablename__ = "corpus_state"

    corpus_name: str = Field(primary_key=True)
    active_run_id: UUID = Field(foreign_key="ingest_run.id")


# --- Data Structure ---

class IngestDocument(SQLModel, table=True):
    """A source file ingested in a specific run."""
    __tablename__ = "ingest_document"

    id: Optional[UUID] = Field(default=None, primary_key=True)
    run_id: UUID = Field(foreign_key="ingest_run.id", index=True)
    
    source_id: str = Field(index=True)  # Stable ID (e.g. filename or URI)
    content_hash: str = Field(index=True) # MD5/SHA256 of content for idempotency check
    
    metadata_json: Optional[dict] = Field(default=None, sa_type=JSON)
    created_at: Optional[datetime] = Field(default=None)

    run: IngestRun = Relationship(back_populates="documents")
    chunks: List["IngestChunk"] = Relationship(back_populates="document")

    __table_args__ = (
        UniqueConstraint("run_id", "source_id", name="uq_run_source"),
    )


class IngestChunk(SQLModel, table=True):
    """A granular text chunk."""
    __tablename__ = "ingest_chunk"

    id: Optional[UUID] = Field(default=None, primary_key=True)
    document_id: UUID = Field(foreign_key="ingest_document.id", index=True)
    
    content: str
    metadata_json: Optional[dict] = Field(default=None, sa_type=JSON) # e.g. {"page": 1, "questions": [...]}
    chunk_index: int
    
    document: IngestDocument = Relationship(back_populates="chunks")
    embedding: Optional["IngestChunkEmbedding"] = Relationship(back_populates="chunk")


class IngestChunkEmbedding(SQLModel, table=True):
    """Vector embedding for a chunk, separated for storage optimization."""
    __tablename__ = "ingest_chunk_embedding"

    chunk_id: UUID = Field(foreign_key="ingest_chunk.id", primary_key=True)
    embedding: List[float] = Field(sa_column=Column(Vector(768))) # gemini-embedding-001 dimension
    
    chunk: IngestChunk = Relationship(back_populates="embedding")
