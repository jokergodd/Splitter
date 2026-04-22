from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


_JSON_SCALAR_TYPES = (str, int, float, bool)


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, _JSON_SCALAR_TYPES):
        return value
    if is_dataclass(value):
        return {item.name: _to_builtin(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Unsupported non-string key for JSON serialization: {key!r}")
            result[key] = _to_builtin(item)
        return result
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-serializable")


@dataclass(slots=True)
class EvalSample:
    sample_id: str
    question: str
    reference_answer: str
    reference_contexts: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(self)


@dataclass(slots=True)
class RetrievalCheckpoint:
    stage_name: str
    child_ids: list[str]
    parent_ids: list[str]
    contexts: list[str]
    query_text: str | None = None
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(self)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    enable_query_rewrite: bool = True
    enable_multi_query_merge: bool = True
    enable_rerank: bool = True
    top_k_per_query: int = 10
    final_parent_limit: int = 5

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(self)


@dataclass(slots=True)
class StageTrace:
    sample: EvalSample
    rewritten_queries: list[str]
    retrieval_checkpoints: list[RetrievalCheckpoint]
    final_answer: str

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(self)
