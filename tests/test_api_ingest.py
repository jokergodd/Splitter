from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingest_service


def _route_by_path(path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route
    raise AssertionError(f"route not found: {path}")


def test_ingest_routes_exist_and_are_async():
    assert inspect.iscoroutinefunction(_route_by_path("/v1/ingest/file").endpoint)
    assert inspect.iscoroutinefunction(_route_by_path("/v1/ingest/batch").endpoint)


def test_ingest_file_uses_injected_service():
    calls: list[tuple[str, str]] = []

    class FakeIngestService:
        async def ingest_file(self, *, file_path: str):
            calls.append(("file", file_path))
            return {"status": "ok", "mode": "file", "file_path": file_path}

        async def ingest_batch(self, *, data_dir: str):
            calls.append(("batch", data_dir))
            return {"status": "ok", "mode": "batch", "data_dir": data_dir}

    app.dependency_overrides[get_ingest_service] = lambda: FakeIngestService()
    client = TestClient(app)

    response = client.post(
        "/v1/ingest/file",
        files={"file": ("doc.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mode"] == "file"
    assert response.json()["file_path"].endswith(".pdf")
    assert len(calls) == 1
    assert calls[0][0] == "file"
    assert calls[0][1].endswith(".pdf")


def test_ingest_batch_uses_injected_service():
    calls: list[tuple[str, str]] = []

    class FakeIngestService:
        async def ingest_file(self, *, file_path: str):
            calls.append(("file", file_path))
            return {"status": "ok", "mode": "file", "file_path": file_path}

        async def ingest_batch(self, *, data_dir: str):
            calls.append(("batch", data_dir))
            return {"status": "ok", "mode": "batch", "data_dir": data_dir}

    app.dependency_overrides[get_ingest_service] = lambda: FakeIngestService()
    client = TestClient(app)

    response = client.post("/v1/ingest/batch", json={"data_dir": "/tmp/data"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "batch", "data_dir": "/tmp/data"}
    assert calls == [("batch", "/tmp/data")]
