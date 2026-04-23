from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, str):
        return {"answer": value}
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {}


class SourceItem(BaseModel):
    parent_id: str | None = None
    source: str | None = None
    file_path: str | None = None


class ChatQueryRequest(BaseModel):
    question: str = Field(min_length=1)


class ChatQueryResponse(BaseModel):
    answer: str
    source_items: list[SourceItem] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: Any) -> "ChatQueryResponse":
        data = _coerce_mapping(result)
        if "answer" not in data and isinstance(data.get("content"), str):
            data["answer"] = data["content"]
        data.setdefault("source_items", [])
        return cls.model_validate(data)


class IngestFileRequest(BaseModel):
    file_path: str = Field(min_length=1)


class IngestBatchRequest(BaseModel):
    data_dir: str = Field(min_length=1)


class IngestResponse(BaseModel):
    status: str = "ok"
    mode: str
    file_path: str | None = None
    data_dir: str | None = None
    detail: str | None = None
    processed_count: int | None = None
    skipped_count: int | None = None
    failed_count: int | None = None

    @classmethod
    def from_result(cls, result: Any, *, mode: str) -> "IngestResponse":
        data = _coerce_mapping(result)
        data.setdefault("mode", mode)
        if "detail" not in data and isinstance(data.get("message"), str):
            data["detail"] = data["message"]
        return cls.model_validate(data)


class TaskSubmissionResponse(BaseModel):
    task_id: str

    @classmethod
    def from_result(cls, result: Any) -> "TaskSubmissionResponse":
        data = _coerce_mapping(result)
        return cls.model_validate(data)


class TaskStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    task_type: str | None = None
    status: str
    progress: float | None = None
    result: Any | None = None
    error: Any | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_result(cls, result: Any, *, task_id: str) -> "TaskStatusResponse":
        data = _coerce_mapping(result)
        data.setdefault("task_id", task_id)
        return cls.model_validate(data)


class StatusResponse(BaseModel):
    status: str
