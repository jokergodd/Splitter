from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

from services.errors import TaskNotFoundError
from services.task_models import TaskStatus, TaskType
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


def test_submit_file_task_delegates_to_ingest_service_and_records_result(tmp_path: Path):
    calls: list[tuple[str, object]] = []

    class FakeIngestService:
        async def ingest_file(self, *, file_path: str, config=None):
            calls.append(("file", file_path))
            return {"status": "ok", "mode": "file", "file_path": file_path}

        async def ingest_batch(self, *, data_dir: str, pipeline_config=None):
            calls.append(("batch", data_dir))
            return {"status": "ok", "mode": "batch", "data_dir": data_dir}

    registry = TaskRegistry()
    service = TaskService(FakeIngestService(), registry=registry)
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    task_id = service.submit_file_task(file_path=file_path)

    task = _wait_for_status(service, task_id, TaskStatus.SUCCEEDED)

    assert task.task_type is TaskType.INGEST_FILE
    assert task.result == {"status": "ok", "mode": "file", "file_path": str(file_path)}
    assert calls == [("file", file_path)]


def test_submit_batch_task_delegates_to_ingest_service_and_records_result(tmp_path: Path):
    calls: list[tuple[str, object]] = []

    class FakeIngestService:
        async def ingest_file(self, *, file_path: str, config=None):
            calls.append(("file", file_path))
            return {"status": "ok", "mode": "file", "file_path": file_path}

        async def ingest_batch(self, *, data_dir: str, pipeline_config=None):
            calls.append(("batch", data_dir))
            return {"status": "ok", "mode": "batch", "data_dir": data_dir}

    registry = TaskRegistry()
    service = TaskService(FakeIngestService(), registry=registry)
    data_dir = tmp_path / "batch"
    data_dir.mkdir()

    task_id = service.submit_batch_task(data_dir=data_dir)

    task = _wait_for_status(service, task_id, TaskStatus.SUCCEEDED)

    assert task.task_type is TaskType.INGEST_BATCH
    assert task.result == {"status": "ok", "mode": "batch", "data_dir": str(data_dir)}
    assert calls == [("batch", data_dir)]


def test_get_task_status_returns_none_for_missing_task():
    service = TaskService(SimpleNamespace())

    assert service.get_task_status("missing") is None


def test_task_service_can_build_ingest_service_from_runtime(monkeypatch):
    runtime = object()
    fake_ingest_service = object()

    monkeypatch.setattr(
        "services.ingest_service.IngestService",
        lambda runtime_arg: fake_ingest_service if runtime_arg is runtime else None,
    )

    service = TaskService(runtime=runtime)

    assert service.ingest_service is fake_ingest_service


def test_task_service_async_api_methods_wrap_sync_submission():
    service = TaskService(SimpleNamespace())
    service.submit_file_task = lambda **kwargs: "task-file-1"  # type: ignore[method-assign]
    service.submit_batch_task = lambda **kwargs: "task-batch-1"  # type: ignore[method-assign]
    service.get_task_status = lambda task_id: {"task_id": task_id, "status": "pending"}  # type: ignore[method-assign]

    async def run():
        file_result = await service.submit_ingest_file(file_path="demo.pdf")
        batch_result = await service.submit_ingest_batch(data_dir="data")
        task_result = await service.get_task(task_id="task-file-1")
        return file_result, batch_result, task_result

    file_result, batch_result, task_result = asyncio.run(run())

    assert file_result == {"task_id": "task-file-1"}
    assert batch_result == {"task_id": "task-batch-1"}
    assert task_result == {"task_id": "task-file-1", "status": "pending"}


def test_get_task_raises_task_not_found_for_missing_task():
    service = TaskService(SimpleNamespace())

    async def run() -> None:
        await service.get_task(task_id="missing")

    try:
        asyncio.run(run())
    except TaskNotFoundError as exc:
        assert exc.task_id == "missing"
    else:
        raise AssertionError("expected TaskNotFoundError")


def test_task_service_close_shuts_down_owned_executor():
    service = TaskService(SimpleNamespace())

    executor = service._executor
    service.close()

    assert getattr(executor, "_shutdown", False) is True
