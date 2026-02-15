"""Retrieval evaluation metrics used by offline runs and CI gates."""

from __future__ import annotations

import math
from typing import Iterable

from evaluation.schemas import GoldChunkRef, RetrievalMetrics


def _norm(values: Iterable[str]) -> list[str]:
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


def _unique_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate while preserving first-seen ranking order."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Precision@k = relevant retrieved in top-k / k."""
    if k <= 0:
        return 0.0
    top_k = _unique_preserve_order(_norm(retrieved_ids[:k]))
    if not top_k:
        return 0.0
    rel = set(_norm(relevant_ids))
    hits = sum(1 for rid in top_k if rid in rel)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@k = relevant retrieved in top-k / all relevant."""
    rel = set(_norm(relevant_ids))
    if not rel:
        return 0.0
    top_k = _unique_preserve_order(_norm(retrieved_ids[:k]))
    hits = len(set(top_k) & rel)
    return hits / len(rel)


def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Reciprocal rank of first relevant result in top-k."""
    rel = set(_norm(relevant_ids))
    if not rel:
        return 0.0
    for idx, rid in enumerate(_unique_preserve_order(_norm(retrieved_ids[:k])), start=1):
        if rid in rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
    graded_relevance: dict[str, float] | None = None,
) -> float:
    """nDCG@k with optional graded relevance."""
    if k <= 0:
        return 0.0

    rel = set(_norm(relevant_ids))
    if not rel:
        return 0.0

    graded = graded_relevance or {rid: 1.0 for rid in rel}

    dcg = 0.0
    for rank, rid in enumerate(_unique_preserve_order(_norm(retrieved_ids[:k])), start=1):
        gain = float(graded.get(rid, 0.0))
        if gain > 0:
            dcg += gain / math.log2(rank + 1)

    ideal_gains = sorted([float(graded.get(rid, 1.0)) for rid in rel], reverse=True)[:k]
    idcg = 0.0
    for rank, gain in enumerate(ideal_gains, start=1):
        idcg += gain / math.log2(rank + 1)

    if idcg <= 0:
        return 0.0
    return dcg / idcg


def build_graded_relevance(
    gold_doc_ids: list[str],
    gold_chunk_refs: list[GoldChunkRef] | list[dict],
) -> dict[str, float]:
    """Build graded doc relevance from doc IDs and chunk references."""
    graded: dict[str, float] = {doc_id: 1.0 for doc_id in _norm(gold_doc_ids)}
    for ref in gold_chunk_refs:
        if isinstance(ref, dict):
            source = ref.get("source", "")
            relevance = float(ref.get("relevance", 1.0))
        else:
            source = ref.source
            relevance = float(ref.relevance)
        if not source:
            continue
        graded[source] = max(graded.get(source, 0.0), min(max(relevance, 0.0), 1.0))
    return graded


def compute_retrieval_metrics(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
    gold_chunk_refs: list[GoldChunkRef] | list[dict],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute all retrieval metrics for one sample."""
    graded = build_graded_relevance(gold_doc_ids, gold_chunk_refs)
    relevant_ids = list(graded.keys())
    retrieved_unique = _unique_preserve_order(_norm(retrieved_doc_ids))

    return RetrievalMetrics(
        k=k,
        precision_at_k=precision_at_k(retrieved_unique, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_unique, relevant_ids, k),
        mrr_at_k=mrr_at_k(retrieved_unique, relevant_ids, k),
        ndcg_at_k=ndcg_at_k(retrieved_unique, relevant_ids, k, graded_relevance=graded),
        retrieved_doc_ids=retrieved_unique[:k],
    )
