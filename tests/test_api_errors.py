from __future__ import annotations

from pymongo.errors import ServerSelectionTimeoutError
from qdrant_client.common.client_exceptions import QdrantException
from qdrant_client.http.exceptions import UnexpectedResponse
from fastapi.testclient import TestClient

from api.app import app
import api.dependencies as api_dependencies
from api.dependencies import get_chat_service, get_ingest_service, get_runtime
from services.exceptions import (
    CollectionNotReadyError,
    DependencyUnavailableError,
    IngestConflictError,
    ModelInitializationError,
    NoContextRetrievedError,
    UnsupportedFileTypeError,
)


def _json_error(response):
    body = response.json()
    assert set(body) == {"code", "message", "details"}
    return body


def test_chat_query_success_is_unchanged():
    class FakeChatService:
        async def answer(self, *, question: str):
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
        "source_items": [{"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    }


def test_value_error_is_mapped_to_bad_request():
    class FakeChatService:
        async def answer(self, *, question: str):
            raise ValueError("question is invalid")

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert _json_error(response) == {
        "code": "BAD_REQUEST",
        "message": "question is invalid",
        "details": {},
    }


def test_unsupported_file_type_domain_error_is_mapped_before_generic_value_error():
    class FakeIngestService:
        async def ingest_file(self, *, file_path: str):
            raise UnsupportedFileTypeError(".csv", supported_types=[".pdf", ".docx"])

    app.dependency_overrides[get_ingest_service] = lambda: FakeIngestService()
    client = TestClient(app)

    response = client.post("/v1/ingest/file", files={"file": ("doc.csv", b"a,b,c", "text/csv")})

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert _json_error(response) == {
        "code": "UNSUPPORTED_FILE_TYPE",
        "message": "Unsupported file type",
        "details": {"file_type": ".csv", "supported_types": [".pdf", ".docx"]},
    }


def test_file_not_found_error_is_mapped_to_not_found():
    class FakeIngestService:
        async def ingest_file(self, *, file_path: str):
            raise FileNotFoundError(file_path)

    app.dependency_overrides[get_ingest_service] = lambda: FakeIngestService()
    client = TestClient(app)

    response = client.post("/v1/ingest/file", files={"file": ("doc.pdf", b"%PDF-1.4\n", "application/pdf")})

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert _json_error(response) == {
        "code": "FILE_NOT_FOUND",
        "message": "File not found",
        "details": {},
    }


def test_not_a_directory_error_is_mapped_to_not_found():
    class FakeIngestService:
        async def ingest_batch(self, *, data_dir: str):
            raise NotADirectoryError(data_dir)

    app.dependency_overrides[get_ingest_service] = lambda: FakeIngestService()
    client = TestClient(app)

    response = client.post("/v1/ingest/batch", json={"data_dir": "/tmp/data"})

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert _json_error(response) == {
        "code": "DIRECTORY_NOT_FOUND",
        "message": "Directory not found",
        "details": {},
    }


def test_validation_error_is_mapped_to_unified_payload():
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": ""})

    assert response.status_code == 422
    body = _json_error(response)
    assert body["code"] == "VALIDATION_ERROR"
    assert body["message"] == "Request validation failed"
    assert body["details"]
    assert body["details"][0]["loc"] == ["body", "question"]


def test_dependency_unavailable_is_mapped_to_service_unavailable():
    def raise_server_selection_timeout():
        raise ServerSelectionTimeoutError("mongo down")

    app.dependency_overrides[get_runtime] = raise_server_selection_timeout
    client = TestClient(app)

    response = client.get("/v1/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert _json_error(response) == {
        "code": "DEPENDENCY_UNAVAILABLE",
        "message": "Required dependency is unavailable",
        "details": {},
    }


def test_model_initialization_error_is_mapped_to_service_unavailable():
    class FakeChatService:
        async def answer(self, *, question: str):
            raise ModelInitializationError("reranker")

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert _json_error(response) == {
        "code": "MODEL_INITIALIZATION_ERROR",
        "message": "Model initialization failed",
        "details": {"component": "reranker"},
    }


def test_qdrant_dependency_unavailable_is_mapped_to_service_unavailable():
    class FakeChatService:
        async def answer(self, *, question: str):
            raise QdrantException("unexpected qdrant failure")

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert _json_error(response) == {
        "code": "DEPENDENCY_UNAVAILABLE",
        "message": "Required dependency is unavailable",
        "details": {},
    }


def test_collection_not_ready_is_mapped_to_service_unavailable():
    class FakeChatService:
        async def answer(self, *, question: str):
            raise CollectionNotReadyError("child_chunks_hybrid")

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert _json_error(response) == {
        "code": "COLLECTION_NOT_READY",
        "message": "Collection 'child_chunks_hybrid' is not ready",
        "details": {},
    }


def test_chat_query_real_dependency_chain_keeps_collection_not_ready_semantics(monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload: dict):
            raise UnexpectedResponse(
                status_code=404,
                reason_phrase="Not Found",
                content=b"{\"status\":{\"error\":\"Not found: Collection `child_chunks_hybrid` doesn't exist!\"}}",
                headers={},
            )

    runtime = type(
        "RuntimeStub",
        (),
        {
            "llm": object(),
            "dense_embeddings": object(),
            "sparse_embeddings": object(),
            "reranker": object(),
            "storage_backend": type(
                "StorageStub",
                (),
                {
                    "qdrant_store": type(
                        "QdrantStub",
                        (),
                        {"async_client": object(), "collection_name": "child_chunks_hybrid"},
                    )(),
                    "mongo_repository": object(),
                },
            )(),
        },
    )()

    monkeypatch.setattr(api_dependencies, "get_runtime", lambda: runtime)
    monkeypatch.setattr("services.chat_graph_service.build_chat_graph", lambda deps: FakeGraph())
    app.state.runtime = None
    app.state.chat_service = None
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.state.runtime = None
    app.state.chat_service = None

    assert response.status_code == 503
    assert _json_error(response) == {
        "code": "COLLECTION_NOT_READY",
        "message": "Collection 'child_chunks_hybrid' is not ready",
        "details": {},
    }


def test_no_context_retrieved_is_mapped_to_not_found():
    class FakeChatService:
        async def answer(self, *, question: str):
            raise NoContextRetrievedError(question)

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert _json_error(response) == {
        "code": "NO_CONTEXT_RETRIEVED",
        "message": "No relevant context was retrieved",
        "details": {},
    }


def test_chat_query_real_dependency_chain_keeps_no_context_semantics(monkeypatch):
    class FakeGraph:
        async def ainvoke(self, payload: dict):
            return {
                "rewritten_queries": ["hello"],
                "parent_chunks": [],
                "response_payload": {
                    "answer": "graph:hello",
                    "source_items": [],
                },
            }

    runtime = type(
        "RuntimeStub",
        (),
        {
            "llm": object(),
            "dense_embeddings": object(),
            "sparse_embeddings": object(),
            "reranker": object(),
            "storage_backend": type(
                "StorageStub",
                (),
                {
                    "qdrant_store": type(
                        "QdrantStub",
                        (),
                        {"async_client": object(), "collection_name": "child_chunks_hybrid"},
                    )(),
                    "mongo_repository": object(),
                },
            )(),
        },
    )()

    monkeypatch.setattr(api_dependencies, "get_runtime", lambda: runtime)
    monkeypatch.setattr("services.chat_graph_service.build_chat_graph", lambda deps: FakeGraph())
    app.state.runtime = None
    app.state.chat_service = None
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.state.runtime = None
    app.state.chat_service = None

    assert response.status_code == 404
    assert _json_error(response) == {
        "code": "NO_CONTEXT_RETRIEVED",
        "message": "No relevant context was retrieved",
        "details": {},
    }


def test_domain_dependency_unavailable_keeps_stable_payload():
    class FakeIngestService:
        async def ingest_file(self, *, file_path: str):
            raise DependencyUnavailableError("MongoDB is unavailable")

    app.dependency_overrides[get_ingest_service] = lambda: FakeIngestService()
    client = TestClient(app)

    response = client.post("/v1/ingest/file", files={"file": ("doc.pdf", b"%PDF-1.4\n", "application/pdf")})

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert _json_error(response) == {
        "code": "DEPENDENCY_UNAVAILABLE",
        "message": "MongoDB is unavailable",
        "details": {},
    }


def test_ingest_conflict_domain_error_is_mapped_to_conflict():
    class FakeIngestService:
        async def ingest_batch(self, *, data_dir: str):
            raise IngestConflictError(content_hash="abc123", reason="already ingested")

    app.dependency_overrides[get_ingest_service] = lambda: FakeIngestService()
    client = TestClient(app)

    response = client.post("/v1/ingest/batch", json={"data_dir": "/tmp/data"})

    app.dependency_overrides.clear()

    assert response.status_code == 409
    assert _json_error(response) == {
        "code": "INGEST_CONFLICT",
        "message": "Ingest request conflicts with existing state",
        "details": {"content_hash": "abc123", "reason": "already ingested"},
    }


def test_unknown_error_is_mapped_to_internal_error():
    class FakeChatService:
        async def answer(self, *, question: str):
            raise RuntimeError("unexpected failure")

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/v1/chat/query", json={"question": "hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 500
    assert _json_error(response) == {
        "code": "INTERNAL_ERROR",
        "message": "Unexpected server error",
        "details": {},
    }
