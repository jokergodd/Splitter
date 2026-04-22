from __future__ import annotations

import pytest

from evals.metrics_generation import (
    build_generation_ragas_rows,
    compute_ragas_generation_metrics,
)
from evals.models import EvalSample, RetrievalCheckpoint, StageTrace


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="sample-1",
        question="Where is the project data stored?",
        reference_answer="The project data is stored in PostgreSQL.",
        reference_contexts=["PostgreSQL stores the authoritative project data."],
    )


def _trace() -> StageTrace:
    return StageTrace(
        sample=_sample(),
        rewritten_queries=["Where is the project data stored?"],
        retrieval_checkpoints=[
            RetrievalCheckpoint(
                stage_name="merged_candidates",
                child_ids=["child-1", "child-2"],
                parent_ids=["parent-1", "parent-2"],
                contexts=["candidate one", "candidate two"],
            ),
            RetrievalCheckpoint(
                stage_name="collapsed_parents",
                child_ids=["child-1"],
                parent_ids=["parent-1"],
                contexts=["PostgreSQL stores the authoritative project data."],
            ),
        ],
        final_answer="The project data is stored in PostgreSQL.",
    )


def test_build_generation_ragas_rows_uses_final_answer_and_checkpoint_contexts():
    rows = build_generation_ragas_rows([_trace()])

    assert rows == [
        {
            "user_input": "Where is the project data stored?",
            "reference": "The project data is stored in PostgreSQL.",
            "response": "The project data is stored in PostgreSQL.",
            "retrieved_contexts": ["PostgreSQL stores the authoritative project data."],
        }
    ]


def test_build_generation_ragas_rows_supports_non_default_checkpoint_stage_name():
    rows = build_generation_ragas_rows([_trace()], checkpoint_stage_name="merged_candidates")

    assert rows == [
        {
            "user_input": "Where is the project data stored?",
            "reference": "The project data is stored in PostgreSQL.",
            "response": "The project data is stored in PostgreSQL.",
            "retrieved_contexts": ["candidate one", "candidate two"],
        }
    ]


def test_compute_ragas_generation_metrics_uses_metric_batch_score_and_factories():
    trace = _trace()
    llm = object()
    embeddings = object()
    calls: dict[str, object] = {"batch_rows": []}

    class _FakeResult:
        def __init__(self, value: float):
            self.value = value

    class _FakeAnswerRelevancyMetric:
        def __init__(self, *, llm, embeddings, name):
            self.llm = llm
            self.embeddings = embeddings
            self.name = name

        def batch_score(self, rows):
            rows = list(rows)
            calls["batch_rows"].append((self.name, rows))
            calls["answer_relevancy_llm"] = self.llm
            calls["answer_relevancy_embeddings"] = self.embeddings
            return [_FakeResult(0.91)]

    class _FakeFaithfulnessMetric:
        def __init__(self, *, llm, name):
            self.llm = llm
            self.name = name

        def batch_score(self, rows):
            rows = list(rows)
            calls["batch_rows"].append((self.name, rows))
            calls["faithfulness_llm"] = self.llm
            return [_FakeResult(0.76)]

    result = compute_ragas_generation_metrics(
        [trace],
        llm=llm,
        embeddings=embeddings,
        answer_relevancy_factory=_FakeAnswerRelevancyMetric,
        faithfulness_factory=_FakeFaithfulnessMetric,
    )

    expected_rows = [
        {
            "user_input": "Where is the project data stored?",
            "reference": "The project data is stored in PostgreSQL.",
            "response": "The project data is stored in PostgreSQL.",
            "retrieved_contexts": ["PostgreSQL stores the authoritative project data."],
        }
    ]
    assert calls["batch_rows"] == [
        ("answer_relevancy", expected_rows),
        ("faithfulness", expected_rows),
    ]
    assert calls["answer_relevancy_llm"] is llm
    assert calls["answer_relevancy_embeddings"] is embeddings
    assert calls["faithfulness_llm"] is llm
    assert result == {
        "answer_relevancy": 0.91,
        "faithfulness": 0.76,
    }


def test_compute_ragas_generation_metrics_rejects_empty_traces():
    with pytest.raises(ValueError, match="traces must not be empty"):
        compute_ragas_generation_metrics(
            [],
            llm=object(),
            embeddings=object(),
        )


def test_compute_ragas_generation_metrics_requires_target_checkpoint():
    trace = StageTrace(
        sample=_sample(),
        rewritten_queries=[],
        retrieval_checkpoints=[],
        final_answer="answer",
    )

    with pytest.raises(ValueError, match="missing retrieval checkpoint"):
        compute_ragas_generation_metrics(
            [trace],
            llm=object(),
            embeddings=object(),
        )
