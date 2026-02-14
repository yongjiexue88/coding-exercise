"""Tests for retrieval metric primitives."""

from evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_basic():
    assert precision_at_k(["a", "b", "c"], ["b", "c"], 3) == 2 / 3


def test_recall_at_k_basic():
    assert recall_at_k(["a", "b", "c"], ["b", "c", "d"], 3) == 2 / 3


def test_mrr_at_k_first_hit():
    assert mrr_at_k(["x", "b", "c"], ["b", "c"], 3) == 0.5


def test_ndcg_at_k_binary():
    score = ndcg_at_k(["a", "b", "c"], ["b", "c"], 3)
    assert 0.0 <= score <= 1.0


def test_compute_retrieval_metrics_from_chunk_refs():
    metrics = compute_retrieval_metrics(
        retrieved_doc_ids=["doc1", "doc2", "doc3"],
        gold_doc_ids=["doc2"],
        gold_chunk_refs=[{"source": "doc3", "relevance": 0.6}],
        k=3,
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr_at_k == 0.5
    assert metrics.ndcg_at_k > 0.0
