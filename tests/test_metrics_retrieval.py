from __future__ import annotations

import pytest

from evals.metrics_retrieval import (
    build_ranking_diagnostics,
    compute_hit_at_k,
    compute_parent_hit_at_k,
    compute_ragas_retrieval_metrics,
)
from evals.models import EvalSample, RetrievalCheckpoint, StageTrace


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="sample-1",
        question="where is the answer stored?",
        reference_answer="The answer is stored in MongoDB.",
        reference_contexts=["MongoDB stores the parent chunks."],
    )


def test_compute_hit_at_k_returns_one_when_reference_is_in_top_k():
    score = compute_hit_at_k(
        retrieved_ids=["child-3", "child-1", "child-2"],
        reference_ids=["child-1"],
        k=2,
    )

    assert score == 1.0


def test_compute_hit_at_k_returns_zero_when_reference_is_outside_top_k():
    score = compute_hit_at_k(
        retrieved_ids=["child-3", "child-1", "child-2"],
        reference_ids=["child-2"],
        k=2,
    )

    assert score == 0.0


def test_compute_parent_hit_at_k_uses_reference_parent_ids():
    score = compute_parent_hit_at_k(
        retrieved_parent_ids=["parent-3", "parent-1", "parent-2"],
        reference_parent_ids=["parent-1"],
        k=2,
    )

    assert score == 1.0


def test_compute_hit_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        compute_hit_at_k(
            retrieved_ids=["child-1"],
            reference_ids=["child-1"],
            k=0,
        )


def test_compute_hit_at_k_returns_zero_for_empty_reference_ids():
    score = compute_hit_at_k(
        retrieved_ids=["child-1", "child-2"],
        reference_ids=[],
        k=1,
    )

    assert score == 0.0


def test_compute_hit_at_k_handles_duplicate_ids_without_double_counting():
    score = compute_hit_at_k(
        retrieved_ids=["child-9", "child-9", "child-1"],
        reference_ids=["child-1", "child-1"],
        k=3,
    )

    assert score == 1.0


def test_build_ranking_diagnostics_reports_failure_mode_signals():
    checkpoint = RetrievalCheckpoint(
        stage_name="merged_candidates",
        child_ids=["child-3", "child-1", "child-1", "child-2"],
        parent_ids=["parent-3", "parent-1", "parent-1", "parent-2"],
        contexts=["c3", "c1", "c2"],
    )

    diagnostics = build_ranking_diagnostics(
        checkpoint,
        reference_child_ids=["child-2", "child-4"],
        reference_parent_ids=["parent-1", "parent-4"],
        ks=(1, 2, 3),
    )

    assert diagnostics["stage_name"] == "merged_candidates"
    assert diagnostics["first_child_hit_rank"] == 4
    assert diagnostics["first_parent_hit_rank"] == 2
    assert diagnostics["child_hit_at_k"] == {1: 0.0, 2: 0.0, 3: 0.0}
    assert diagnostics["parent_hit_at_k"] == {1: 0.0, 2: 1.0, 3: 1.0}
    assert diagnostics["retrieved_child_count"] == 4
    assert diagnostics["retrieved_parent_count"] == 4
    assert diagnostics["unique_child_count"] == 3
    assert diagnostics["unique_parent_count"] == 3
    assert diagnostics["reference_child_count"] == 2
    assert diagnostics["reference_parent_count"] == 2
    assert diagnostics["matched_child_ids"] == ["child-2"]
    assert diagnostics["matched_parent_ids"] == ["parent-1"]
    assert diagnostics["missing_child_reference_ids"] == ["child-4"]
    assert diagnostics["missing_parent_reference_ids"] == ["parent-4"]
    assert diagnostics["top_child_ids_without_reference_hit"] == ["child-3", "child-1", "child-1"]
    assert diagnostics["top_parent_ids_without_reference_hit"] == ["parent-3"]
    assert diagnostics["context_count"] == 3


def test_build_ranking_diagnostics_rejects_non_positive_k():
    checkpoint = RetrievalCheckpoint(
        stage_name="merged_candidates",
        child_ids=["child-1"],
        parent_ids=["parent-1"],
        contexts=["c1"],
    )

    with pytest.raises(ValueError, match="k must be positive"):
        build_ranking_diagnostics(checkpoint, ks=(0,))


def test_compute_ragas_retrieval_metrics_uses_collapsed_parent_contexts():
    trace = StageTrace(
        sample=_sample(),
        rewritten_queries=["where is the answer stored?"],
        retrieval_checkpoints=[
            RetrievalCheckpoint(
                stage_name="merged_candidates",
                child_ids=["child-1", "child-2"],
                parent_ids=["parent-1", "parent-2"],
                contexts=["child one", "child two"],
            ),
            RetrievalCheckpoint(
                stage_name="collapsed_parents",
                child_ids=["child-1"],
                parent_ids=["parent-1"],
                contexts=["MongoDB stores the parent chunks."],
            ),
        ],
        final_answer="The answer is stored in MongoDB.",
    )
    calls: list[tuple[str, list[dict[str, object]]]] = []

    class _FakeResult:
        def __init__(self, value: float):
            self.value = value

    class _FakeMetric:
        def __init__(self, *, llm, name):
            self.llm = llm
            self.name = name

        def batch_score(self, rows):
            rows = list(rows)
            calls.append((self.name, rows))
            value = 0.75 if self.name == "context_precision" else 0.5
            return [_FakeResult(value)]

    result = compute_ragas_retrieval_metrics(
        [trace],
        llm=object(),
        context_precision_factory=_FakeMetric,
        context_recall_factory=_FakeMetric,
    )

    expected_rows = [
        {
            "user_input": "where is the answer stored?",
            "reference": "The answer is stored in MongoDB.",
            "retrieved_contexts": ["MongoDB stores the parent chunks."],
        }
    ]
    assert calls == [
        ("context_precision", expected_rows),
        ("context_recall", expected_rows),
    ]
    assert result == {
        "context_precision": 0.75,
        "context_recall": 0.5,
    }


def test_compute_ragas_retrieval_metrics_supports_non_default_checkpoint_stage_name():
    trace = StageTrace(
        sample=_sample(),
        rewritten_queries=["where is the answer stored?"],
        retrieval_checkpoints=[
            RetrievalCheckpoint(
                stage_name="merged_candidates",
                child_ids=["child-1", "child-2"],
                parent_ids=["parent-1", "parent-2"],
                contexts=["child one", "child two"],
            ),
            RetrievalCheckpoint(
                stage_name="collapsed_parents",
                child_ids=["child-1"],
                parent_ids=["parent-1"],
                contexts=["parent one"],
            ),
        ],
        final_answer="The answer is stored in MongoDB.",
    )
    calls: list[tuple[str, list[dict[str, object]]]] = []

    class _FakeResult:
        def __init__(self, value: float):
            self.value = value

    class _FakeMetric:
        def __init__(self, *, llm, name):
            self.llm = llm
            self.name = name

        def batch_score(self, rows):
            rows = list(rows)
            calls.append((self.name, rows))
            value = 0.1 if self.name == "context_precision" else 0.2
            return [_FakeResult(value)]

    result = compute_ragas_retrieval_metrics(
        [trace],
        llm=object(),
        checkpoint_stage_name="merged_candidates",
        context_precision_factory=_FakeMetric,
        context_recall_factory=_FakeMetric,
    )

    expected_rows = [
        {
            "user_input": "where is the answer stored?",
            "reference": "The answer is stored in MongoDB.",
            "retrieved_contexts": ["child one", "child two"],
        }
    ]
    assert calls == [
        ("context_precision", expected_rows),
        ("context_recall", expected_rows),
    ]
    assert result == {
        "context_precision": 0.1,
        "context_recall": 0.2,
    }


def test_compute_ragas_retrieval_metrics_supports_multiple_checkpoint_stage_names():
    trace = StageTrace(
        sample=_sample(),
        rewritten_queries=["where is the answer stored?"],
        retrieval_checkpoints=[
            RetrievalCheckpoint(
                stage_name="merged_candidates",
                child_ids=["child-1", "child-2"],
                parent_ids=["parent-1", "parent-2"],
                contexts=["child one", "child two"],
            ),
            RetrievalCheckpoint(
                stage_name="collapsed_parents",
                child_ids=["child-1"],
                parent_ids=["parent-1"],
                contexts=["parent one"],
            ),
        ],
        final_answer="The answer is stored in MongoDB.",
    )
    seen_calls: list[tuple[str, list[dict[str, object]]]] = []

    class _FakeResult:
        def __init__(self, value: float):
            self.value = value

    class _FakeMetric:
        def __init__(self, *, llm, name):
            self.llm = llm
            self.name = name

        def batch_score(self, rows):
            rows = list(rows)
            seen_calls.append((self.name, rows))
            if rows[0]["retrieved_contexts"] == ["child one", "child two"]:
                value = 0.3 if self.name == "context_precision" else 0.4
            else:
                value = 0.8 if self.name == "context_precision" else 0.9
            return [_FakeResult(value)]

    result = compute_ragas_retrieval_metrics(
        [trace],
        llm=object(),
        checkpoint_stage_name=("merged_candidates", "collapsed_parents"),
        context_precision_factory=_FakeMetric,
        context_recall_factory=_FakeMetric,
    )

    assert seen_calls == [
        (
            "context_precision",
            [
                {
                    "user_input": "where is the answer stored?",
                    "reference": "The answer is stored in MongoDB.",
                    "retrieved_contexts": ["child one", "child two"],
                }
            ],
        ),
        (
            "context_recall",
            [
                {
                    "user_input": "where is the answer stored?",
                    "reference": "The answer is stored in MongoDB.",
                    "retrieved_contexts": ["child one", "child two"],
                }
            ],
        ),
        (
            "context_precision",
            [
                {
                    "user_input": "where is the answer stored?",
                    "reference": "The answer is stored in MongoDB.",
                    "retrieved_contexts": ["parent one"],
                }
            ],
        ),
        (
            "context_recall",
            [
                {
                    "user_input": "where is the answer stored?",
                    "reference": "The answer is stored in MongoDB.",
                    "retrieved_contexts": ["parent one"],
                }
            ],
        ),
    ]
    assert result == {
        "merged_candidates": {
            "context_precision": 0.3,
            "context_recall": 0.4,
        },
        "collapsed_parents": {
            "context_precision": 0.8,
            "context_recall": 0.9,
        },
    }


def test_compute_ragas_retrieval_metrics_rejects_empty_traces():
    with pytest.raises(ValueError, match="traces must not be empty"):
        compute_ragas_retrieval_metrics([], llm=object())


def test_compute_ragas_retrieval_metrics_requires_target_checkpoint():
    trace = StageTrace(
        sample=_sample(),
        rewritten_queries=[],
        retrieval_checkpoints=[],
        final_answer="answer",
    )

    with pytest.raises(ValueError, match="missing retrieval checkpoint"):
        compute_ragas_retrieval_metrics([trace], llm=object())
