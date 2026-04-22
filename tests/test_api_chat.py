from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_chat_service


def _route_by_path(path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route
    raise AssertionError(f"route not found: {path}")


def test_chat_query_route_exists_and_is_async():
    assert inspect.iscoroutinefunction(_route_by_path("/v1/chat/query").endpoint)


def test_chat_query_uses_injected_service(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeChatService:
        async def answer(self, *, question: str):
            calls.append({"question": question})
            return {
                "answer": f"echo:{question}",
                "source_items": [{"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
            }

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "echo:hello",
        "source_items": [
            {"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}
        ],
    }
    assert calls == [{"question": "hello"}]
