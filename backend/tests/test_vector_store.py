"""Tests for the vector store service."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import chromadb


class TestVectorStore:
    """Tests for ChromaDB vector store operations using a fresh in-memory client."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Create a fresh ChromaDB client for each test."""
        self.client = chromadb.PersistentClient(path=str(tmp_path / "test_chroma"))
        self.collection = self.client.get_or_create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"},
        )

    def test_add_and_search(self):
        """Test adding documents and searching by embedding."""
        self.collection.upsert(
            ids=["1", "2", "3"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            documents=["Python is great", "JavaScript is fun", "CSS is styling"],
            metadatas=[
                {"source": "python.md"},
                {"source": "js.md"},
                {"source": "css.md"},
            ],
        )

        results = self.collection.query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )
        assert len(results["documents"][0]) == 2
        assert "Python is great" in results["documents"][0]

    def test_document_count(self):
        """Test document count tracking."""
        assert self.collection.count() == 0
        self.collection.upsert(
            ids=["1"],
            embeddings=[[1.0, 0.0]],
            documents=["test"],
            metadatas=[{"source": "test.md"}],
        )
        assert self.collection.count() == 1

    def test_get_all_sources(self):
        """Test retrieving source document metadata."""
        self.collection.upsert(
            ids=["1", "2", "3"],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            documents=["chunk 1", "chunk 2", "chunk 3"],
            metadatas=[
                {"source": "doc_a.md"},
                {"source": "doc_a.md"},
                {"source": "doc_b.md"},
            ],
        )
        all_data = self.collection.get(include=["metadatas"])
        source_counts = {}
        for meta in all_data["metadatas"]:
            src = meta.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        assert source_counts["doc_a.md"] == 2
        assert source_counts["doc_b.md"] == 1

    def test_reset(self):
        """Test collection reset clears all data."""
        self.collection.upsert(
            ids=["1"],
            embeddings=[[1.0]],
            documents=["test"],
            metadatas=[{"source": "test.md"}],
        )
        assert self.collection.count() == 1
        self.client.delete_collection("test_collection")
        new_collection = self.client.get_or_create_collection(name="test_collection")
        assert new_collection.count() == 0
