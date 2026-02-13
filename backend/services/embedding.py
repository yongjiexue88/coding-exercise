"""Embedding service using Google Gemini."""

from __future__ import annotations

import google.generativeai as genai
from config import settings


class EmbeddingService:
    """Service for generating text embeddings using Gemini text-embedding-004."""

    def __init__(self):
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self.model_name = "models/gemini-embedding-001"
        self.dimension = 768

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document",
            output_dimensionality=self.dimension,
        )
        return result["embedding"]

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query."""
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=self.dimension,
        )
        return result["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_document",
            output_dimensionality=self.dimension,
        )
        return result["embedding"]
