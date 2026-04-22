from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any

from ragas.metrics.collections.context_precision import ContextPrecisionWithReference
from ragas.metrics.collections.context_recall import ContextRecall

from evals.models import RetrievalCheckpoint, StageTrace


def _validate_k(k: int) -> int:
    if k <= 0:
        raise ValueError("k must be positive")
    return k


def _normalize_ids(ids: Sequence[str]) -> list[str]:
    return [str(item) for item in ids]


def _unique_ids(ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(_normalize_ids(ids)))


def _first_hit_rank(retrieved_ids: Sequence[str], reference_ids: Sequence[str]) -> int | None:
    reference_set = set(_normalize_ids(reference_ids))
    if not reference_set:
        return None

    for rank, item in enumerate(_normalize_ids(retrieved_ids), start=1):
        if item in reference_set:
            return rank
    return None


def _ids_before_first_hit(retrieved_ids: Sequence[str], reference_ids: Sequence[str]) -> list[str]:
    first_hit_rank = _first_hit_rank(retrieved_ids, reference_ids)
    normalized = _normalize_ids(retrieved_ids)
    if first_hit_rank is None:
        return normalized
    return normalized[: first_hit_rank - 1]


def _matched_reference_ids(retrieved_ids: Sequence[str], reference_ids: Sequence[str]) -> list[str]:
    retrieved_unique = set(_unique_ids(retrieved_ids))
    return [item for item in _unique_ids(reference_ids) if item in retrieved_unique]


def _missing_reference_ids(retrieved_ids: Sequence[str], reference_ids: Sequence[str]) -> list[str]:
    retrieved_unique = set(_unique_ids(retrieved_ids))
    return [item for item in _unique_ids(reference_ids) if item not in retrieved_unique]


def compute_hit_at_k(retrieved_ids: list[str], reference_ids: list[str], k: int) -> float:
    _validate_k(k)
    reference_set = set(_normalize_ids(reference_ids))
    if not reference_set:
        return 0.0
    return 1.0 if any(item in reference_set for item in _normalize_ids(retrieved_ids)[:k]) else 0.0


def compute_parent_hit_at_k(
    retrieved_parent_ids: list[str],
    reference_parent_ids: list[str],
    k: int,
) -> float:
    return compute_hit_at_k(retrieved_parent_ids, reference_parent_ids, k)


def build_ranking_diagnostics(
    checkpoint: RetrievalCheckpoint,
    *,
    reference_child_ids: Sequence[str] = (),
    reference_parent_ids: Sequence[str] = (),
    ks: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    normalized_ks = tuple(_validate_k(k) for k in ks)
    child_ids = _normalize_ids(checkpoint.child_ids)
    parent_ids = _normalize_ids(checkpoint.parent_ids)
    unique_child_ids = _unique_ids(checkpoint.child_ids)
    unique_parent_ids = _unique_ids(checkpoint.parent_ids)
    unique_reference_child_ids = _unique_ids(reference_child_ids)
    unique_reference_parent_ids = _unique_ids(reference_parent_ids)

    return {
        "stage_name": checkpoint.stage_name,
        "context_count": len(checkpoint.contexts),
        "retrieved_child_count": len(child_ids),
        "retrieved_parent_count": len(parent_ids),
        "unique_child_count": len(unique_child_ids),
        "unique_parent_count": len(unique_parent_ids),
        "reference_child_count": len(unique_reference_child_ids),
        "reference_parent_count": len(unique_reference_parent_ids),
        "first_child_hit_rank": _first_hit_rank(child_ids, unique_reference_child_ids),
        "first_parent_hit_rank": _first_hit_rank(parent_ids, unique_reference_parent_ids),
        "child_hit_at_k": {
            k: compute_hit_at_k(child_ids, unique_reference_child_ids, k)
            for k in normalized_ks
        },
        "parent_hit_at_k": {
            k: compute_parent_hit_at_k(parent_ids, unique_reference_parent_ids, k)
            for k in normalized_ks
        },
        "matched_child_ids": _matched_reference_ids(child_ids, unique_reference_child_ids),
        "matched_parent_ids": _matched_reference_ids(parent_ids, unique_reference_parent_ids),
        "missing_child_reference_ids": _missing_reference_ids(
            child_ids,
            unique_reference_child_ids,
        ),
        "missing_parent_reference_ids": _missing_reference_ids(
            parent_ids,
            unique_reference_parent_ids,
        ),
        "top_child_ids_without_reference_hit": _ids_before_first_hit(
            child_ids,
            unique_reference_child_ids,
        ),
        "top_parent_ids_without_reference_hit": _ids_before_first_hit(
            parent_ids,
            unique_reference_parent_ids,
        ),
    }


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


def _build_ragas_rows(
    traces: Sequence[StageTrace],
    *,
    checkpoint_stage_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        checkpoint = _get_checkpoint(trace, checkpoint_stage_name=checkpoint_stage_name)
        rows.append(
            {
                "user_input": trace.sample.question,
                "reference": trace.sample.reference_answer,
                "retrieved_contexts": list(checkpoint.contexts),
            }
        )
    return rows


def _normalize_checkpoint_stage_names(
    checkpoint_stage_name: str | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(checkpoint_stage_name, str):
        return (checkpoint_stage_name,)

    normalized = tuple(str(item) for item in checkpoint_stage_name)
    if not normalized:
        raise ValueError("checkpoint_stage_name must not be empty")
    return normalized


def _compute_single_stage_metrics(
    traces: Sequence[StageTrace],
    *,
    llm: Any,
    checkpoint_stage_name: str,
    context_precision_factory: Any,
    context_recall_factory: Any,
) -> dict[str, float]:
    rows = _build_ragas_rows(traces, checkpoint_stage_name=checkpoint_stage_name)
    metrics = {
        "context_precision": context_precision_factory(
            llm=llm,
            name="context_precision",
        ),
        "context_recall": context_recall_factory(
            llm=llm,
            name="context_recall",
        ),
    }
    return {
        metric_name: fmean(float(result.value) for result in metric.batch_score(rows))
        for metric_name, metric in metrics.items()
    }


def compute_ragas_retrieval_metrics(
    traces: Sequence[StageTrace],
    *,
    llm: Any,
    checkpoint_stage_name: str | Sequence[str] = "collapsed_parents",
    context_precision_factory: Any = ContextPrecisionWithReference,
    context_recall_factory: Any = ContextRecall,
) -> dict[str, float] | dict[str, dict[str, float]]:
    if not traces:
        raise ValueError("traces must not be empty")

    stage_names = _normalize_checkpoint_stage_names(checkpoint_stage_name)
    per_stage_results = {
        stage_name: _compute_single_stage_metrics(
            traces,
            llm=llm,
            checkpoint_stage_name=stage_name,
            context_precision_factory=context_precision_factory,
            context_recall_factory=context_recall_factory,
        )
        for stage_name in stage_names
    }
    if len(stage_names) == 1:
        return per_stage_results[stage_names[0]]
    return per_stage_results


__all__ = [
    "build_ranking_diagnostics",
    "compute_hit_at_k",
    "compute_parent_hit_at_k",
    "compute_ragas_retrieval_metrics",
]
