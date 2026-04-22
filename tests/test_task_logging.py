from __future__ import annotations

import time
from pathlib import Path

from services.task_models import TaskStatus
from services.task_registry import TaskRegistry
from services.task_service import TaskService


def _wait_for_status(service: TaskService, task_id: str, status: TaskStatus, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = service.get_task_status(task_id)
        if task is not None and task.status is status:
            return task
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach {status}")


def test_task_service_logs_submit_start_and_success(caplog, tmp_path: Path):
    class FakeIngestService:
        async def ingest_file(self, *, file_path: str, config=None):
            return {"status": "ok", "file_path": file_path}

        async def ingest_batch(self, *, data_dir: str, pipeline_config=None):
            return {"status": "ok", "data_dir": data_dir}

    service = TaskService(FakeIngestService(), registry=TaskRegistry())
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    with caplog.at_level("INFO"):
        task_id = service.submit_file_task(file_path=file_path)
        _wait_for_status(service, task_id, TaskStatus.SUCCEEDED)

    events = [record.event for record in caplog.records if getattr(record, "task_id", None) == task_id]
    assert events == [
        "task.submitted",
        "task.started",
        "task.succeeded",
    ]
    submitted = next(record for record in caplog.records if getattr(record, "event", None) == "task.submitted")
    succeeded = next(record for record in caplog.records if getattr(record, "event", None) == "task.succeeded")
    assert submitted.task_type == "ingest_file"
    assert submitted.file_path == str(file_path)
    assert succeeded.duration_ms >= 0


def test_task_service_logs_failure(caplog, tmp_path: Path):
    class FakeIngestService:
        async def ingest_file(self, *, file_path: str, config=None):
            raise RuntimeError("boom")

        async def ingest_batch(self, *, data_dir: str, pipeline_config=None):
            raise RuntimeError("boom")

    service = TaskService(FakeIngestService(), registry=TaskRegistry())
    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    with caplog.at_level("INFO"):
        task_id = service.submit_file_task(file_path=file_path)
        _wait_for_status(service, task_id, TaskStatus.FAILED)

    events = [record.event for record in caplog.records if getattr(record, "task_id", None) == task_id]
    assert events == [
        "task.submitted",
        "task.started",
        "task.failed",
    ]
    failed = next(record for record in caplog.records if getattr(record, "event", None) == "task.failed")
    assert failed.error == "boom"
