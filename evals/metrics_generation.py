from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean
from typing import Any

from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.faithfulness import Faithfulness

from evals.models import RetrievalCheckpoint, StageTrace


def _get_checkpoint(
    trace: StageTrace,
    *,
    checkpoint_stage_name: str,
) -> RetrievalCheckpoint:
    for checkpoint in trace.retrieval_checkpoints:
        if checkpoint.stage_name == checkpoint_stage_name:
            return checkpoint
    raise ValueError(
        f"StageTrace for sample {trace.sample.sample_id!r} is missing retrieval checkpoint "
        f"{checkpoint_stage_name!r}"
    )


def build_generation_ragas_rows(
    traces: Sequence[StageTrace],
    *,
    checkpoint_stage_name: str = "collapsed_parents",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        checkpoint = _get_checkpoint(trace, checkpoint_stage_name=checkpoint_stage_name)
        rows.append(
            {
                "user_input": trace.sample.question,
                "reference": trace.sample.reference_answer,
                "response": trace.final_answer,
                "retrieved_contexts": list(checkpoint.contexts),
            }
        )
    return rows


def compute_ragas_generation_metrics(
    traces: Sequence[StageTrace],
    *,
    llm: Any,
    embeddings: Any,
    checkpoint_stage_name: str = "collapsed_parents",
    answer_relevancy_factory: Any = AnswerRelevancy,
    faithfulness_factory: Any = Faithfulness,
) -> dict[str, float]:
    if not traces:
        raise ValueError("traces must not be empty")

    rows = build_generation_ragas_rows(
        traces,
        checkpoint_stage_name=checkpoint_stage_name,
    )
    metrics = {
        "answer_relevancy": answer_relevancy_factory(
            llm=llm,
            embeddings=embeddings,
            name="answer_relevancy",
        ),
        "faithfulness": faithfulness_factory(
            llm=llm,
            name="faithfulness",
        ),
    }
    return {
        metric_name: fmean(float(result.value) for result in metric.batch_score(rows))
        for metric_name, metric in metrics.items()
    }


__all__ = [
    "build_generation_ragas_rows",
    "compute_ragas_generation_metrics",
]
