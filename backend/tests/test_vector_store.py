"""Tests for the SQLModel-backed vector store service."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from models_ingest import CorpusState, IngestChunk, IngestChunkEmbedding, IngestDocument, IngestRun
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
        ("chunk-1", 0, "content1", "doc1.json", {"title": "Doc1"}, 0.1),
        ("chunk-2", 1, "content2", "doc2.json", None, 0.2),
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
    # mock .first() returning a Row-like tuple
    session.exec.return_value.first.return_value = (42,)

    store = VectorStoreService()
    count = store.get_document_count()

    assert count == 42


@patch.object(VectorStoreService, "_ensure_extensions", return_value=None)
@patch("services.vector_store.Session")
def test_get_document_count_handles_none(mock_session_cls, _mock_init):
    """Count query should return 0 if no results."""
    session = mock_session_cls.return_value.__enter__.return_value
    session.exec.return_value.first.return_value = None

    store = VectorStoreService()
    count = store.get_document_count()

    assert count == 0


@patch.object(VectorStoreService, "_ensure_extensions", return_value=None)
@patch("services.vector_store.Session")
def test_get_all_sources_formats_rows(mock_session_cls, _mock_init):
    """Source listing should map SQL rows to API response format."""
    session = mock_session_cls.return_value.__enter__.return_value
    session.exec.return_value.fetchall.return_value = [
        ("SQuAD-small.json", 128),
        ("extra.json", 12),
    ]

    store = VectorStoreService()
    sources = store.get_all_sources()

    assert sources == [
        {"source": "SQuAD-small.json", "chunk_count": 128},
        {"source": "extra.json", "chunk_count": 12},
    ]


@patch.object(VectorStoreService, "_ensure_extensions", return_value=None)
@patch("services.vector_store.Session")
def test_reset_creates_new_active_run(mock_session_cls, _mock_init):
    """Legacy reset should rotate corpus_state to a fresh run."""
    session = mock_session_cls.return_value.__enter__.return_value
    previous_run_id = uuid4()
    state = CorpusState(corpus_name="default", active_run_id=previous_run_id)
    previous_run = IngestRun(id=previous_run_id, corpus_name="default", status="ready")

    def get_side_effect(model_cls, key):
        if model_cls is CorpusState and key == "default":
            return state
        if model_cls is IngestRun and key == previous_run_id:
            return previous_run
        return None

    session.get.side_effect = get_side_effect

    store = VectorStoreService()
    store.reset()

    assert previous_run.status == "archived"
    added = [call.args[0] for call in session.add.call_args_list]
    assert any(isinstance(obj, IngestRun) and obj.id != previous_run_id for obj in added)
    assert any(isinstance(obj, CorpusState) for obj in added)
    session.commit.assert_called_once()


@patch.object(VectorStoreService, "_ensure_extensions", return_value=None)
@patch("services.vector_store.Session")
def test_add_documents_legacy_compat_inserts_rows(mock_session_cls, _mock_init):
    """Legacy add_documents API should still ingest chunks + embeddings."""
    session = mock_session_cls.return_value.__enter__.return_value
    active_run_id = uuid4()
    state = CorpusState(corpus_name="default", active_run_id=active_run_id)
    active_run = IngestRun(id=active_run_id, corpus_name="default", status="ready")

    def get_side_effect(model_cls, key):
        if model_cls is CorpusState and key == "default":
            return state
        if model_cls is IngestRun and key == active_run_id:
            return active_run
        return None

    session.get.side_effect = get_side_effect
    session.exec.return_value.first.return_value = None

    store = VectorStoreService()
    store.add_documents(
        ids=["legacy-chunk-id"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Legacy content"],
        metadatas=[{"source": "legacy.md", "chunk_index": "7"}],
    )

    added = [call.args[0] for call in session.add.call_args_list]
    inserted_doc = next(obj for obj in added if isinstance(obj, IngestDocument))
    inserted_chunk = next(obj for obj in added if isinstance(obj, IngestChunk))
    inserted_embedding = next(obj for obj in added if isinstance(obj, IngestChunkEmbedding))

    assert inserted_doc.source_id == "legacy.md"
    assert inserted_chunk.chunk_index == 7
    assert inserted_chunk.metadata_json["legacy_id"] == "legacy-chunk-id"
    assert inserted_embedding.embedding == [0.1, 0.2, 0.3]
    session.commit.assert_called_once()
