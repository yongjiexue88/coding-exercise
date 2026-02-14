"""Tests for user-query input guards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import QueryRequest
from services.query_guard import normalize_and_validate_query


def test_normalize_and_validate_query_collapses_whitespace() -> None:
    query = " \n\tWhat   is   RAG?\t "
    assert normalize_and_validate_query(query) == "What is RAG?"


def test_normalize_and_validate_query_rejects_whitespace_only() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        normalize_and_validate_query("  \n\t   ")


def test_query_request_normalizes_input_before_downstream_use() -> None:
    req = QueryRequest(query="  Explain   FastAPI\narchitecture  ")
    assert req.query == "Explain FastAPI architecture"


def test_query_request_rejects_whitespace_only_query() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="\n\t  ")
