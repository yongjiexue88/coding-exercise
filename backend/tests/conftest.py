import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_db_connection():
    """Mock database connection globally to prevent integration attempts."""
    with patch("psycopg2.connect") as mock_connect, \
         patch("services.vector_store.register_vector") as mock_register, \
         patch("services.rag.VectorStoreService") as mock_vss:
        
        # Setup mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Setup VectorStoreService mock to behave like a dict-like store if needed, 
        # or just let it return None for methods to avoid errors in simple instantiations.
        # But for RAG service tests, we might want FakeVectorStoreService to take precedence.
        # Since RAG tests use dependency injection, this generic patch mainly protects 
        # API endpoints that might instantiate default services.
        
        yield mock_connect
