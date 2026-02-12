"""Tests for the evaluation metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from evaluation.metrics import (
    context_precision,
    context_recall_at_k,
    keyword_coverage,
    answer_length_score,
    faithfulness_heuristic,
)


class TestContextPrecision:
    def test_source_found(self):
        assert context_precision(["a.md", "b.md", "c.md"], "b.md") == 1.0

    def test_source_not_found(self):
        assert context_precision(["a.md", "b.md"], "c.md") == 0.0

    def test_empty_retrieved(self):
        assert context_precision([], "a.md") == 0.0


class TestContextRecallAtK:
    def test_first_position(self):
        assert context_recall_at_k(["a.md", "b.md"], "a.md") == 1.0

    def test_second_position(self):
        assert context_recall_at_k(["a.md", "b.md"], "b.md") == 0.5

    def test_not_found(self):
        assert context_recall_at_k(["a.md", "b.md"], "c.md") == 0.0


class TestKeywordCoverage:
    def test_all_keywords_found(self):
        answer = "Flexbox uses justify-content and align-items for centering"
        assert keyword_coverage(answer, ["flexbox", "centering"]) == 1.0

    def test_partial_coverage(self):
        answer = "Flexbox is great for layouts"
        assert keyword_coverage(answer, ["flexbox", "centering"]) == 0.5

    def test_empty_keywords(self):
        assert keyword_coverage("any answer", []) == 1.0


class TestAnswerLengthScore:
    def test_ideal_length(self):
        answer = " ".join(["word"] * 100)
        assert answer_length_score(answer) == 1.0

    def test_too_short(self):
        answer = "short"
        score = answer_length_score(answer)
        assert score < 1.0

    def test_too_long(self):
        answer = " ".join(["word"] * 1000)
        score = answer_length_score(answer)
        assert score < 1.0


class TestFaithfulness:
    def test_high_faithfulness(self):
        context = ["Python is a great programming language for data science and machine learning"]
        answer = "Python is a great programming language used for data science."
        score = faithfulness_heuristic(answer, context)
        assert score >= 0.5

    def test_empty_answer(self):
        score = faithfulness_heuristic("", ["some context"])
        assert score == 1.0
