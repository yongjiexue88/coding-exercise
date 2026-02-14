"""Legacy-compatible metric helpers.

These wrappers are kept for backward compatibility while the V1 evaluation
runner migrates to retrieval_metrics.py + judges.py + perf_metrics.py.
"""

from evaluation.answer_metrics import answer_length_score as _answer_length_score
from evaluation.retrieval_metrics import mrr_at_k


def context_precision(retrieved_sources: list[str], expected_source: str) -> float:
    """Measure if the expected source document was retrieved.

    Returns 1.0 if expected source is in retrieved sources, 0.0 otherwise.
    Higher score = better retrieval precision.
    """
    return 1.0 if expected_source in retrieved_sources else 0.0


def context_recall_at_k(retrieved_sources: list[str], expected_source: str) -> float:
    """Measure the position of the expected source in retrieved results.

    Returns a score between 0.0 and 1.0, where 1.0 means the expected
    source was the first result.
    """
    return mrr_at_k(retrieved_sources, [expected_source], k=max(len(retrieved_sources), 1))


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """Measure what fraction of expected keywords appear in the answer.

    Returns a score between 0.0 and 1.0.
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def answer_length_score(answer: str, min_words: int = 20, max_words: int = 500) -> float:
    """Score the answer length — penalize too short or too long answers.

    Returns 1.0 for answers in the ideal range, lower for extremes.
    """
    return _answer_length_score(answer, min_words=min_words, max_words=max_words)


def faithfulness_heuristic(answer: str, context_docs: list[str]) -> float:
    """Heuristic faithfulness check — measures overlap between answer and context.

    Counts what fraction of answer sentences have significant word overlap
    with the context. This is a simple heuristic, not a full NLI check.
    """
    import re

    context_text = " ".join(context_docs).lower()
    context_words = set(context_text.split())

    # Remove very common words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "neither", "each", "every", "all", "any", "few",
        "more", "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when", "while",
        "this", "that", "these", "those", "it", "its", "you", "your", "i",
    }
    context_words -= stop_words

    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return 1.0

    grounded_count = 0
    for sentence in sentences:
        sentence_words = set(sentence.lower().split()) - stop_words
        if not sentence_words:
            continue
        overlap = len(sentence_words & context_words) / len(sentence_words)
        if overlap > 0.3:  # 30% word overlap threshold
            grounded_count += 1

    return grounded_count / len(sentences) if sentences else 1.0
