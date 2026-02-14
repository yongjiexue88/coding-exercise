"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with lifespan context."""
    from main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_check(self, client):
        """Test health endpoint returns correct structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_model" in data
        assert "gemini" in data["active_model"]
        assert "documents_indexed" in data


class TestQueryEndpoint:
    def test_query_without_documents(self, client):
        """Test that query fails gracefully with no indexed documents."""
        response = client.post("/query", json={"query": "test query"})
        assert response.status_code in [200, 400]

    def test_query_validation(self, client):
        """Test request validation rejects empty query."""
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_query_validation_rejects_whitespace_only_query(self, client):
        """Test request validation rejects whitespace-only query."""
        response = client.post("/query", json={"query": "   \n\t  "})
        assert response.status_code == 422


class TestDocumentsEndpoint:
    def test_list_documents(self, client):
        """Test listing indexed documents."""
        response = client.get("/documents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint is exposed."""
        response = client.get("/metrics")
        assert response.status_code == 200


class TestQueryStreamEndpoint:
    def test_query_stream_emits_status_chunk_done_sequence(self, client, monkeypatch):
        """Streaming endpoint should emit status events before chunk/done and keep done fields compatible."""
        import main

        class FakeSource:
            def model_dump(self):
                return {"content": "Doc content", "source": "doc.md", "relevance_score": 0.91}

        class FakeVectorStore:
            def get_document_count(self):
                return 1

        class FakeRAGService:
            def __init__(self):
                self.vector_store = FakeVectorStore()
                self.last_observation = {"route": "rag_simple", "validator_decision": "pass"}

            async def query_stream(self, query, top_k=3, progress_callback=None):
                if progress_callback:
                    await progress_callback(
                        {
                            "step": "understand",
                            "state": "started",
                            "label": "Understanding question",
                        }
                    )
                    await progress_callback(
                        {
                            "step": "understand",
                            "state": "completed",
                            "label": "Understanding question",
                        }
                    )

                async def fake_stream():
                    yield "Hello world"

                return fake_stream(), [FakeSource()], "gemini/fake-model", 12.0

        monkeypatch.setattr(main, "rag_service", FakeRAGService())

        response = client.post("/query/stream", json={"query": "hello", "top_k": 3})
        assert response.status_code == 200

        payloads = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [payload["type"] for payload in payloads] == ["status", "status", "chunk", "done"]

        done_payload = payloads[-1]
        assert "sources" in done_payload
        assert "model" in done_payload
        assert "query_time_ms" in done_payload
        assert done_payload["quality_summary"] == {
            "verification": "verified",
            "sources_used": 1,
        }


class TestCorsBehavior:
    def test_unhandled_server_error_includes_cors_header(self):
        """Unhandled 500 responses should still include CORS headers for allowed origins."""
        from main import app as asgi_app

        inner_app = asgi_app.app if hasattr(asgi_app, "app") else asgi_app
        test_path = "/__cors_error_test"

        @inner_app.get(test_path)
        async def _cors_error_test():
            raise RuntimeError("intentional cors test crash")

        with TestClient(asgi_app, raise_server_exceptions=False) as c:
            response = c.get(test_path, headers={"Origin": "https://yongjiexue.com"})

        assert response.status_code == 500
        assert response.headers.get("access-control-allow-origin") == "https://yongjiexue.com"
