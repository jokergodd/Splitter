from __future__ import annotations

import inspect
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api.app import app
import api.dependencies as api_dependencies
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
        async def answer(self, *, question: str, request_id: str | None = None):
            calls.append({"question": question, "request_id": request_id})
            return {
                "answer": f"echo:{question}",
                "source_items": [{"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
            }

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"}, headers={"X-Request-ID": "req-456"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "echo:hello",
        "source_items": [
            {"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}
        ],
    }
    assert calls == [{"question": "hello", "request_id": "req-456"}]


def test_chat_query_keeps_compatibility_with_services_without_request_id(monkeypatch) -> None:
    class LegacyChatService:
        async def answer(self, *, question: str):
            return {"answer": f"legacy:{question}", "source_items": []}

    app.dependency_overrides[get_chat_service] = lambda: LegacyChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"}, headers={"X-Request-ID": "req-789"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"answer": "legacy:hello", "source_items": []}


def test_chat_query_supports_async_dependency_override() -> None:
    class AsyncChatService:
        async def answer(self, *, question: str, request_id: str | None = None):
            return {"answer": f"async:{question}:{request_id}", "source_items": []}

    async def override_chat_service():
        return AsyncChatService()

    app.dependency_overrides[get_chat_service] = override_chat_service
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"}, headers={"X-Request-ID": "req-999"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"answer": "async:hello:req-999", "source_items": []}


def test_chat_query_uses_real_chat_graph_service_dependency_chain(monkeypatch) -> None:
    build_calls: list[object] = []
    invoke_calls: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        llm=object(),
        dense_embeddings=object(),
        sparse_embeddings=object(),
        reranker=object(),
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(async_client="async-client", collection_name="collection"),
            mongo_repository="mongo",
        ),
    )

    class FakeGraph:
        async def ainvoke(self, payload: dict):
            invoke_calls.append(payload)
            return {
                "parent_chunks": [
                    Document(
                        page_content="context",
                        metadata={"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"},
                    )
                ],
                "response_payload": {
                    "answer": f"graph:{payload['question']}",
                    "source_items": [
                        {"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}
                    ],
                }
            }

    def fake_build_chat_graph(deps):
        build_calls.append(deps)
        return FakeGraph()

    monkeypatch.setattr(api_dependencies, "get_runtime", lambda: runtime)
    monkeypatch.setattr("services.chat_graph_service.build_chat_graph", fake_build_chat_graph)
    app.state.runtime = None
    if hasattr(app.state, "chat_service"):
        app.state.chat_service = None
    client = TestClient(app)

    first_response = client.post("/v1/chat/query", json={"question": "hello"}, headers={"X-Request-ID": "req-1"})
    second_response = client.post("/v1/chat/query", json={"question": "again"}, headers={"X-Request-ID": "req-2"})

    if hasattr(app.state, "chat_service"):
        app.state.chat_service = None
    app.state.runtime = None

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json() == {
        "answer": "graph:hello",
        "source_items": [{"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    }
    assert second_response.json() == {
        "answer": "graph:again",
        "source_items": [{"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    }
    assert len(build_calls) == 1
    assert invoke_calls == [
        {
            "question": "hello",
            "top_k": 10,
            "candidate_limit": 30,
            "max_queries": 4,
            "parent_limit": 5,
            "request_id": "req-1",
        },
        {
            "question": "again",
            "top_k": 10,
            "candidate_limit": 30,
            "max_queries": 4,
            "parent_limit": 5,
            "request_id": "req-2",
        },
    ]


def test_chat_query_validation_does_not_force_runtime_initialization(monkeypatch) -> None:
    app.state.runtime = None
    if hasattr(app.state, "chat_service"):
        app.state.chat_service = None
    monkeypatch.setattr(api_dependencies, "get_runtime", lambda: (_ for _ in ()).throw(AssertionError("runtime should stay lazy")))

    client = TestClient(app)
    response = client.post("/v1/chat/query", json={"question": ""})

    if hasattr(app.state, "chat_service"):
        app.state.chat_service = None
    app.state.runtime = None

    assert response.status_code == 422


def test_instantiate_service_reraises_constructor_type_error() -> None:
    class BrokenService:
        def __init__(self, runtime):
            raise TypeError("constructor exploded")

    try:
        api_dependencies._instantiate_service(BrokenService, runtime=object())
    except TypeError as exc:
        assert str(exc) == "constructor exploded"
    else:
        raise AssertionError("expected constructor TypeError to propagate")
