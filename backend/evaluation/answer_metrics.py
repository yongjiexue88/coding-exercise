"""Non-judge answer metrics used in evaluation reports."""

from __future__ import annotations


def answer_length_score(answer: str, min_words: int = 20, max_words: int = 500) -> float:
    """Score answer length in [0, 1], penalizing very short and very long outputs."""
    word_count = len(answer.split())
    if word_count < min_words:
        return word_count / min_words if min_words > 0 else 1.0
    if word_count > max_words:
        return max(0.5, max_words / max(word_count, 1))
    return 1.0
