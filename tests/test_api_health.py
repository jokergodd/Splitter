from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_runtime


def _route_by_path(path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route
    raise AssertionError(f"route not found: {path}")


def test_health_routes_exist_and_are_async():
    assert inspect.iscoroutinefunction(_route_by_path("/v1/health").endpoint)
    assert inspect.iscoroutinefunction(_route_by_path("/v1/ready").endpoint)


def test_health_endpoint_returns_ok():
    client = TestClient(app)

    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_returns_ready():
    class FakeMongoAdmin:
        async def command(self, name: str):
            assert name == "ping"
            return {"ok": 1}

    class FakeMongoClient:
        admin = FakeMongoAdmin()

    class FakeQdrantClient:
        async def get_collections(self):
            return {"collections": []}

    runtime = type(
        "Runtime",
        (),
        {
            "storage_backend": type(
                "StorageBackend",
                (),
                {
                    "mongo_repository": type(
                        "MongoRepository",
                        (),
                        {"async_client": FakeMongoClient()},
                    )(),
                    "qdrant_store": type(
                        "QdrantStore",
                        (),
                        {"async_client": FakeQdrantClient()},
                    )(),
                },
            )(),
        },
    )()
    app.dependency_overrides[get_runtime] = lambda: runtime
    client = TestClient(app)

    response = client.get("/v1/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
