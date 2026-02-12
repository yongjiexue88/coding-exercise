"""ChromaDB vector store service."""

from __future__ import annotations

import chromadb
from config import settings


class VectorStoreService:
    """Service for managing document embeddings in ChromaDB."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add document chunks with embeddings to the vector store."""
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self, query_embedding: list[float], top_k: int = 3
    ) -> dict:
        """Search for similar documents using a query embedding.

        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances' keys.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_document_count(self) -> int:
        """Get the total number of chunks in the collection."""
        return self.collection.count()

    def get_all_sources(self) -> list[dict]:
        """Get a summary of all indexed document sources."""
        all_data = self.collection.get(include=["metadatas"])
        source_counts: dict[str, int] = {}
        for meta in all_data["metadatas"]:
            source = meta.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        return [
            {"source": source, "chunk_count": count}
            for source, count in sorted(source_counts.items())
        ]

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
