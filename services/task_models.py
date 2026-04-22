from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskType(str, Enum):
    INGEST_FILE = "ingest_file"
    INGEST_BATCH = "ingest_batch"


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_type: TaskType
    status: TaskStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    result: Any = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


__all__ = ["TaskRecord", "TaskStatus", "TaskType", "utc_now"]
