"""Tests for the SQLModel-backed vector store service."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.vector_store import VectorStoreService


@patch("services.vector_store.Session")
def test_ensure_extensions_creates_pgvector(mock_session_cls):
    """Service init should attempt to enable pgvector extension."""
    session = mock_session_cls.return_value.__enter__.return_value

    VectorStoreService()

    assert session.exec.call_count == 1
    stmt = session.exec.call_args[0][0]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in str(stmt)
    session.commit.assert_called_once()


@patch.object(VectorStoreService, "_ensure_extensions", return_value=None)
@patch("services.vector_store.Session")
def test_search_formats_result_shape(mock_session_cls, _mock_init):
    """Search should return documents/metadatas/distances shape expected by RAG."""
    session = mock_session_cls.return_value.__enter__.return_value
    exec_result = MagicMock()
    exec_result.fetchall.return_value = [
        ("content1", "doc1.json", {"title": "Doc1"}, 0.1),
        ("content2", "doc2.json", None, 0.2),
    ]
    session.exec.return_value = exec_result

    store = VectorStoreService()
    results = store.search([0.1, 0.2], top_k=2)

    assert results["documents"][0] == ["content1", "content2"]
    assert results["distances"][0] == [0.1, 0.2]
    assert results["metadatas"][0][0]["source"] == "doc1.json"
    assert results["metadatas"][0][1]["source"] == "doc2.json"


@patch.object(VectorStoreService, "_ensure_extensions", return_value=None)
@patch("services.vector_store.Session")
def test_get_document_count_from_active_run(mock_session_cls, _mock_init):
    """Count query should return scalar count for active corpus run."""
    session = mock_session_cls.return_value.__enter__.return_value
    session.exec.return_value.one.return_value = 42

    store = VectorStoreService()
    count = store.get_document_count()

    assert count == 42


@patch.object(VectorStoreService, "_ensure_extensions", return_value=None)
@patch("services.vector_store.Session")
def test_get_all_sources_formats_rows(mock_session_cls, _mock_init):
    """Source listing should map SQL rows to API response format."""
    session = mock_session_cls.return_value.__enter__.return_value
    session.exec.return_value.fetchall.return_value = [
        ("SQuAD-v1.1.json", 128),
        ("extra.json", 12),
    ]

    store = VectorStoreService()
    sources = store.get_all_sources()

    assert sources == [
        {"source": "SQuAD-v1.1.json", "chunk_count": 128},
        {"source": "extra.json", "chunk_count": 12},
    ]
