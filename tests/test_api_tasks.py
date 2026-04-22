from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_task_service
from services.errors import TaskNotFoundError


def _route_by_path(path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route
    raise AssertionError(f"route not found: {path}")


def test_task_routes_exist_and_are_async():
    assert inspect.iscoroutinefunction(_route_by_path("/v1/tasks/ingest/file").endpoint)
    assert inspect.iscoroutinefunction(_route_by_path("/v1/tasks/ingest/batch").endpoint)
    assert inspect.iscoroutinefunction(_route_by_path("/v1/tasks/{task_id}").endpoint)


def test_task_ingest_file_returns_task_id():
    calls: list[tuple[str, str]] = []

    class FakeTaskService:
        async def submit_ingest_file(self, *, file_path: str):
            calls.append(("file", file_path))
            return {"task_id": "task-file-1"}

        async def submit_ingest_batch(self, *, data_dir: str):
            calls.append(("batch", data_dir))
            return {"task_id": "task-batch-1"}

        async def get_task(self, *, task_id: str):
            calls.append(("get", task_id))
            return {"task_id": task_id, "task_type": "ingest_file", "status": "pending"}

    app.dependency_overrides[get_task_service] = lambda: FakeTaskService()
    client = TestClient(app)

    response = client.post(
        "/v1/tasks/ingest/file",
        files={"file": ("doc.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-file-1"}
    assert len(calls) == 1
    assert calls[0][0] == "file"
    assert calls[0][1].endswith(".pdf")


def test_task_ingest_batch_returns_task_id():
    calls: list[tuple[str, str]] = []

    class FakeTaskService:
        async def submit_ingest_file(self, *, file_path: str):
            calls.append(("file", file_path))
            return {"task_id": "task-file-1"}

        async def submit_ingest_batch(self, *, data_dir: str):
            calls.append(("batch", data_dir))
            return {"task_id": "task-batch-1"}

        async def get_task(self, *, task_id: str):
            calls.append(("get", task_id))
            return {"task_id": task_id, "task_type": "ingest_batch", "status": "pending"}

    app.dependency_overrides[get_task_service] = lambda: FakeTaskService()
    client = TestClient(app)

    response = client.post("/v1/tasks/ingest/batch", json={"data_dir": "/tmp/data"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-batch-1"}
    assert calls == [("batch", "/tmp/data")]


def test_task_lookup_returns_structured_status():
    class FakeTaskService:
        async def submit_ingest_file(self, *, file_path: str):
            return {"task_id": "task-file-1"}

        async def submit_ingest_batch(self, *, data_dir: str):
            return {"task_id": "task-batch-1"}

        async def get_task(self, *, task_id: str):
            return {
                "task_id": task_id,
                "task_type": "ingest_file",
                "status": "running",
                "progress": 0.5,
                "result": None,
                "error": None,
            }

    app.dependency_overrides[get_task_service] = lambda: FakeTaskService()
    client = TestClient(app)

    response = client.get("/v1/tasks/task-123")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "task-123",
        "task_type": "ingest_file",
        "status": "running",
        "progress": 0.5,
        "result": None,
        "error": None,
        "created_at": None,
        "started_at": None,
        "finished_at": None,
    }


def test_missing_task_returns_task_not_found():
    class FakeTaskService:
        async def submit_ingest_file(self, *, file_path: str):
            return {"task_id": "task-file-1"}

        async def submit_ingest_batch(self, *, data_dir: str):
            return {"task_id": "task-batch-1"}

        async def get_task(self, *, task_id: str):
            raise TaskNotFoundError(task_id)

    app.dependency_overrides[get_task_service] = lambda: FakeTaskService()
    client = TestClient(app)

    response = client.get("/v1/tasks/missing-task")

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json() == {
        "code": "TASK_NOT_FOUND",
        "message": "Task not found",
        "details": {},
    }
