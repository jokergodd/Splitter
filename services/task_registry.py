from __future__ import annotations

from threading import Lock
from uuid import uuid4

from services.task_models import TaskRecord, TaskStatus, TaskType, utc_now


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = Lock()

    def register(self, task_type: TaskType) -> TaskRecord:
        task = TaskRecord(
            task_id=uuid4().hex,
            task_type=task_type,
            status=TaskStatus.PENDING,
            progress=0.0,
            created_at=utc_now(),
        )
        with self._lock:
            self._tasks[task.task_id] = task
        return task.model_copy(deep=True)

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return None if task is None else task.model_copy(deep=True)

    def list(self) -> list[TaskRecord]:
        with self._lock:
            return [task.model_copy(deep=True) for task in self._tasks.values()]

    def mark_running(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TaskStatus.RUNNING
            task.progress = max(task.progress, 0.0)
            if task.started_at is None:
                task.started_at = utc_now()
            return task.model_copy(deep=True)

    def mark_succeeded(self, task_id: str, result) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TaskStatus.SUCCEEDED
            task.progress = 1.0
            task.result = result
            if task.started_at is None:
                task.started_at = utc_now()
            task.finished_at = utc_now()
            task.error = None
            return task.model_copy(deep=True)

    def mark_failed(self, task_id: str, error: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TaskStatus.FAILED
            task.progress = 1.0
            task.error = error
            if task.started_at is None:
                task.started_at = utc_now()
            task.finished_at = utc_now()
            return task.model_copy(deep=True)


__all__ = ["TaskRegistry"]
