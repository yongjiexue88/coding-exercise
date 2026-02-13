"""Neon PostgreSQL vector store service."""

from __future__ import annotations

import json
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from config import settings


class VectorStoreService:
    """Service for managing document embeddings in Neon PostgreSQL."""

    def __init__(self):
        self.conn = psycopg2.connect(settings.database_url)
        self.conn.autocommit = True
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database schema and extensions."""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table for document chunks
            # Use 768 dimensions to match EmbeddingService output_dimensionality.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT,
                    content TEXT,
                    embedding vector(768),
                    source TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create HNSW index for faster similarity search
            # This is optional but recommended for performance as data grows
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embedding_idx 
                ON document_chunks 
                USING hnsw (embedding vector_cosine_ops)
            """)
            
        register_vector(self.conn)

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add document chunks with embeddings to the vector store."""
        with self.conn.cursor() as cur:
            for i in range(len(ids)):
                cur.execute(
                    """
                    INSERT INTO document_chunks (doc_id, content, embedding, source, metadata) 
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        ids[i],
                        documents[i],
                        embeddings[i],
                        metadatas[i].get("source", "unknown"),
                        Json(metadatas[i])
                    )
                )

    def search(
        self, query_embedding: list[float], top_k: int = 3
    ) -> dict:
        """Search for similar documents using a query embedding.

        Returns retrieval results in the shape expected by RAGService.
        """
        with self.conn.cursor() as cur:
            # Using <=> operator for cosine distance (0 = identical, 1 = opposite)
            cur.execute("""
                SELECT content, source, metadata, embedding <=> %s::vector as distance
                FROM document_chunks
                ORDER BY distance ASC
                LIMIT %s
            """, (query_embedding, top_k))
            
            rows = cur.fetchall()
            
            if not rows:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            # Keep a stable response format for downstream RAG processing.
            return {
                "documents": [[r[0] for r in rows]],
                "metadatas": [[r[2] for r in rows]],
                "distances": [[float(r[3]) for r in rows]]
            }

    def get_document_count(self) -> int:
        """Get the total number of chunks in the collection."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            return cur.fetchone()[0]

    def get_all_sources(self) -> list[dict]:
        """Get a summary of all indexed document sources."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT source, COUNT(*) as chunk_count
                FROM document_chunks
                GROUP BY source
                ORDER BY source
            """)
            rows = cur.fetchall()
            
        return [
            {"source": row[0], "chunk_count": row[1]}
            for row in rows
        ]

    def reset(self) -> None:
        """Delete all data in the table."""
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE document_chunks")
