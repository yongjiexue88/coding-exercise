"""Tests for the vector store service."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from services.vector_store import VectorStoreService

class TestVectorStore:
    """Tests for VectorStoreService using mocks for psycopg2."""

    @pytest.fixture
    def mock_conn(self):
        with patch("psycopg2.connect") as mock_connect, \
             patch("services.vector_store.register_vector") as mock_register:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            yield mock_conn, mock_cursor

    def test_initialize_db(self, mock_conn):
        """Test database initialization."""
        conn, cursor = mock_conn
        VectorStoreService()
        
        # Check if table creation SQL was executed
        assert cursor.execute.call_count >= 1
        ensure_table = any("CREATE TABLE IF NOT EXISTS document_chunks" in str(args) for args, _ in cursor.execute.call_args_list)
        assert ensure_table

    def test_add_documents(self, mock_conn):
        """Test adding documents."""
        conn, cursor = mock_conn
        store = VectorStoreService()
        
        ids = ["1", "2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        documents = ["doc1", "doc2"]
        metadatas = [{"source": "s1"}, {"source": "s2"}]
        
        store.add_documents(ids, embeddings, documents, metadatas)
        
        # Verify insert execution
        assert cursor.execute.call_count >= 1
        insert_calls = [
            args for args, _ in cursor.execute.call_args_list 
            if "INSERT INTO document_chunks" in str(args[0])
        ]
        assert len(insert_calls) == 2

    def test_search(self, mock_conn):
        """Test search query."""
        conn, cursor = mock_conn
        store = VectorStoreService()
        
        # Mock search result
        cursor.fetchall.return_value = [
            ("content1", "s1", {"source": "s1"}, 0.1),
            ("content2", "s2", {"source": "s2"}, 0.2)
        ]
        
        results = store.search([0.1, 0.2], top_k=2)
        
        assert len(results["documents"][0]) == 2
        assert results["documents"][0][0] == "content1"
        assert results["distances"][0][0] == 0.1

