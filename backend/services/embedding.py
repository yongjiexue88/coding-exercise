"""Embedding service using sentence-transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


class EmbeddingService:
    """Service for generating text embeddings using all-MiniLM-L6-v2."""

    def __init__(self):
        self.model = _get_model()
        self.dimension = 384

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
        )
        return embeddings.tolist()
