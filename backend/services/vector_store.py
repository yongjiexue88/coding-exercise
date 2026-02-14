"""Neon PostgreSQL vector store service using SQLModel and PgVector."""

from __future__ import annotations

from typing import List, Optional
from sqlalchemy import text
from sqlmodel import Session, select
from models_ingest import IngestChunk, IngestChunkEmbedding, IngestDocument, CorpusState
from database import engine

class VectorStoreService:
    """Service for managing document embeddings in Neon PostgreSQL via SQLModel."""

    def __init__(self):
        self._ensure_extensions()

    def _ensure_extensions(self):
        """Ensure pgvector extension exists."""
        with Session(engine) as session:
            session.exec(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        corpus_name: str = "default"
    ) -> dict:
        """Search for similar chunks in the *active* ingestion run.

        Performs a join:
        Embedding -> Chunk -> Document -> Run (via CorpusState active_run_id)
        """
        with Session(engine) as session:
            # SQLModel doesn't support vector operators in pure python syntax easily yet,
            # so we use a text() construct for the cosine distance ordering.
            # We select strictly from the active run for this corpus.

            stmt = text("""
                SELECT
                    c.id,
                    c.chunk_index,
                    c.content,
                    d.source_id,
                    c.metadata_json,
                    e.embedding <=> :embedding AS distance
                FROM ingest_chunk_embedding e
                JOIN ingest_chunk c ON e.chunk_id = c.id
                JOIN ingest_document d ON c.document_id = d.id
                JOIN corpus_state cs ON d.run_id = cs.active_run_id
                WHERE cs.corpus_name = :corpus_name
                ORDER BY distance ASC
                LIMIT :top_k
            """)

            results = session.exec(
                stmt,
                params={
                    "embedding": str(query_embedding), # pgvector expects string representation or array
                    "corpus_name": corpus_name,
                    "top_k": top_k
                }
            ).fetchall()

            # Format results for RAGService
            if not results:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            current_docs = []
            current_metas = []
            current_dists = []

            for row in results:
                chunk_id, chunk_index, content, source, meta, dist = row
                current_docs.append(content)
                # Merge explicit source_id with other metadata
                if meta is None: meta = {}
                meta["source"] = source
                meta["chunk_id"] = str(chunk_id)
                meta["chunk_index"] = chunk_index
                current_metas.append(meta)
                current_dists.append(float(dist))

            return {
                "documents": [current_docs],
                "metadatas": [current_metas],
                "distances": [current_dists]
            }

    def get_document_count(self, corpus_name: str = "default") -> int:
        """Get the total number of chunks in the active run."""
        with Session(engine) as session:
            stmt = text("""
                SELECT COUNT(*)
                FROM ingest_chunk c
                JOIN ingest_document d ON c.document_id = d.id
                JOIN corpus_state cs ON d.run_id = cs.active_run_id
                WHERE cs.corpus_name = :corpus_name
            """)
            return session.exec(stmt, params={"corpus_name": corpus_name}).one()[0]

    def get_all_sources(self, corpus_name: str = "default") -> List[dict]:
        """Get a summary of all indexed sources in the active run."""
        with Session(engine) as session:
            stmt = text("""
                SELECT d.source_id, COUNT(*) as chunk_count
                FROM ingest_chunk c
                JOIN ingest_document d ON c.document_id = d.id
                JOIN corpus_state cs ON d.run_id = cs.active_run_id
                WHERE cs.corpus_name = :corpus_name
                GROUP BY d.source_id
                ORDER BY d.source_id
            """)
            results = session.exec(stmt, params={"corpus_name": corpus_name}).fetchall()

            return [
                {"source": row[0], "chunk_count": row[1]}
                for row in results
            ]
