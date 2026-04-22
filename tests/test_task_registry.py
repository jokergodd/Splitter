from __future__ import annotations

from services.task_registry import TaskRegistry
from services.task_models import TaskStatus, TaskType


def test_register_task_creates_pending_record():
    registry = TaskRegistry()

    task = registry.register(TaskType.INGEST_FILE)

    assert task.task_id
    assert task.task_type is TaskType.INGEST_FILE
    assert task.status is TaskStatus.PENDING
    assert task.progress == 0.0
    assert task.result is None
    assert task.error is None
    assert task.created_at is not None


def test_registry_updates_task_state_through_success():
    registry = TaskRegistry()
    task = registry.register(TaskType.INGEST_BATCH)

    running = registry.mark_running(task.task_id)
    succeeded = registry.mark_succeeded(task.task_id, {"status": "ok"})

    assert running is not None
    assert running.status is TaskStatus.RUNNING
    assert running.started_at is not None
    assert succeeded is not None
    assert succeeded.status is TaskStatus.SUCCEEDED
    assert succeeded.finished_at is not None
    assert succeeded.progress == 1.0
    assert succeeded.result == {"status": "ok"}
    assert registry.get(task.task_id) == succeeded


def test_registry_marks_task_failed_and_stores_error():
    registry = TaskRegistry()
    task = registry.register(TaskType.INGEST_FILE)

    failed = registry.mark_failed(task.task_id, "boom")

    assert failed is not None
    assert failed.status is TaskStatus.FAILED
    assert failed.finished_at is not None
    assert failed.progress == 1.0
    assert failed.error == "boom"
