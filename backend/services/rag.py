"""RAG pipeline orchestration service."""

from __future__ import annotations

import time
from typing import AsyncIterator
from services.embedding import EmbeddingService
from services.vector_store import VectorStoreService
from services.llm import LLMService
from models import SourceDocument


class RAGService:
    """Orchestrates the full RAG pipeline: embed → retrieve → generate."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()

    async def query(self, query: str, top_k: int = 3) -> dict:
        """Execute the full RAG pipeline and return a complete response.

        Returns:
            Dict with 'answer', 'sources', 'model', and 'query_time_ms'.
        """
        start_time = time.time()

        # Step 1: Embed the query
        query_embedding = self.embedding_service.embed_text(query)

        # Step 2: Retrieve relevant documents
        results = self.vector_store.search(query_embedding, top_k=top_k)
        context_docs = self._format_results(results)
        sources = self._build_sources(results)

        # Step 3: Generate response using LLM
        answer = await self.llm_service.generate(query, context_docs)

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "model": f"gemini/{self.llm_service.current_model_name}",
            "query_time_ms": round(elapsed_ms, 2),
        }

    async def query_stream(
        self, query: str, top_k: int = 3
    ) -> tuple[AsyncIterator[str], list[SourceDocument], str]:
        """Execute RAG pipeline with streaming LLM response.

        Returns:
            Tuple of (text_stream, sources, model_name).
        """
        # Step 1: Embed the query
        query_embedding = self.embedding_service.embed_text(query)

        # Step 2: Retrieve relevant documents
        results = self.vector_store.search(query_embedding, top_k=top_k)
        context_docs = self._format_results(results)
        sources = self._build_sources(results)
        model_name = f"gemini/{self.llm_service.current_model_name}"

        # Step 3: Create streaming generator
        stream = self.llm_service.generate_stream(query, context_docs)

        return stream, sources, model_name

    def _format_results(self, results: dict) -> list[dict]:
        """Format ChromaDB results into context documents for the LLM."""
        docs = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                docs.append({
                    "content": doc,
                    "source": metadata.get("source", "Unknown"),
                })
        return docs

    def _build_sources(self, results: dict) -> list[SourceDocument]:
        """Build SourceDocument list from ChromaDB results."""
        sources = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert cosine distance to similarity score (1 - distance)
                relevance = round(1 - distance, 4)
                sources.append(
                    SourceDocument(
                        content=doc,
                        source=metadata.get("source", "Unknown"),
                        relevance_score=relevance,
                    )
                )
        return sources
