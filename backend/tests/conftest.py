import os
import pytest
from unittest.mock import MagicMock, patch


# Set dummy DATABASE_URL for tests if not present, to avoid ValueError in database.py
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/testdb"


async def _noop_run_loop(self):
    return None


@pytest.fixture(autouse=True)
def mock_db_connection():
    """Mock runtime dependencies for tests that should avoid real DB side effects."""
    mock_store = MagicMock()
    mock_store.get_document_count.return_value = 0
    mock_store.get_all_sources.return_value = []
    mock_store.search.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    with patch("main.create_db_and_tables"), \
         patch("services.rag.VectorStoreService", return_value=mock_store), \
         patch("data.pipeline.jobs.JobWorker.run_loop", new=_noop_run_loop):
        yield mock_store
