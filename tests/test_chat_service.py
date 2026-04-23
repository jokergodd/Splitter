from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from langchain_core.documents import Document
from pymongo import errors as pymongo_errors
from qdrant_client.common.client_exceptions import QdrantException
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_demo.answering import AnswerResult
from services import chat_service
from services.errors import CollectionNotReadyError, DependencyUnavailableError, NoContextRetrievedError


def test_chat_service_delegates_to_chat_graph_service(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatGraphService:
        def __init__(self, runtime):
            captured["runtime"] = runtime

        async def answer(self, **kwargs):
            captured["kwargs"] = kwargs
            return AnswerResult(
                answer="delegated",
                parent_chunks=[
                    Document(
                        page_content="context",
                        metadata={"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"},
                    )
                ],
                source_items=[{"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
            )

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    service = chat_service.ChatService(SimpleNamespace())
    result = asyncio.run(service.answer(question="hello", top_k=2))

    assert result == AnswerResult(
        answer="delegated",
        parent_chunks=[
            Document(
                page_content="context",
                metadata={"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"},
            )
        ],
        source_items=[{"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    )
    assert captured == {
        "runtime": service.runtime,
        "kwargs": {
            "question": "hello",
            "top_k": 2,
            "candidate_limit": 30,
            "max_queries": 4,
            "parent_limit": 5,
        },
    }


def test_chat_service_raises_no_context_retrieved_for_empty_source_items(monkeypatch, caplog) -> None:
    class FakeChatGraphService:
        def __init__(self, runtime):
            self.runtime = runtime

        async def answer(self, **kwargs):
            return {"answer": "delegated", "source_items": []}

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    service = chat_service.ChatService(SimpleNamespace())

    with caplog.at_level(logging.INFO):
        try:
            asyncio.run(service.answer(question="hello"))
        except NoContextRetrievedError as exc:
            assert exc.question == "hello"
        else:
            raise AssertionError("expected NoContextRetrievedError")

    events = [record.event for record in caplog.records if getattr(record, "event", None) and record.name == chat_service.__name__]
    assert "chat.answer.started" in events
    assert "chat.answer.failed" in events
    failed = next(record for record in caplog.records if getattr(record, "event", None) == "chat.answer.failed")
    assert failed.question == "hello"
    assert failed.error == "No relevant context was retrieved"


def test_module_answer_delegates_to_chat_graph_service(monkeypatch) -> None:
    runtime = SimpleNamespace()
    captured: dict[str, object] = {}

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            captured["runtime"] = runtime_arg

        async def answer(self, **kwargs):
            captured["kwargs"] = kwargs
            return {
                "answer": "module-delegated",
                "source_items": [{"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
            }

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    result = asyncio.run(chat_service.answer("hello", runtime, candidate_limit=7))

    assert result == {
        "answer": "module-delegated",
        "source_items": [{"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    }
    assert captured == {
        "runtime": runtime,
        "kwargs": {
            "question": "hello",
            "top_k": 10,
            "candidate_limit": 7,
            "max_queries": 4,
            "parent_limit": 5,
        },
    }


def test_module_answer_wraps_missing_collection_as_collection_not_ready(monkeypatch) -> None:
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(collection_name="child_chunks_hybrid"),
        ),
    )

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            raise UnexpectedResponse(
                status_code=404,
                reason_phrase="Not Found",
                content=b"{\"status\":{\"error\":\"Not found: Collection `child_chunks_hybrid` doesn't exist!\"}}",
                headers={},
            )

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.answer("hello", runtime))
    except CollectionNotReadyError as exc:
        assert exc.collection_name == "child_chunks_hybrid"
    else:
        raise AssertionError("expected CollectionNotReadyError")


def test_module_answer_raises_no_context_retrieved_for_empty_source_items(monkeypatch) -> None:
    runtime = SimpleNamespace()

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            return {"answer": "module-delegated", "source_items": []}

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.answer("hello", runtime))
    except NoContextRetrievedError as exc:
        assert exc.question == "hello"
    else:
        raise AssertionError("expected NoContextRetrievedError")


def test_chat_service_wraps_missing_collection_as_collection_not_ready(monkeypatch) -> None:
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(collection_name="child_chunks_hybrid"),
        ),
    )

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            raise UnexpectedResponse(
                status_code=404,
                reason_phrase="Not Found",
                content=b"{\"status\":{\"error\":\"Not found: Collection `child_chunks_hybrid` doesn't exist!\"}}",
                headers={},
            )

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.ChatService(runtime).answer(question="hello"))
    except CollectionNotReadyError as exc:
        assert exc.collection_name == "child_chunks_hybrid"
    else:
        raise AssertionError("expected CollectionNotReadyError")


def test_module_answer_wraps_qdrant_errors_as_dependency_unavailable(monkeypatch) -> None:
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(collection_name="collection"),
        ),
    )

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            raise QdrantException("qdrant timeout")

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.answer("hello", runtime))
    except DependencyUnavailableError as exc:
        assert exc.dependency == "qdrant"
        assert str(exc) == "Qdrant is unavailable"
    else:
        raise AssertionError("expected DependencyUnavailableError")


def test_chat_service_wraps_qdrant_errors_as_dependency_unavailable(monkeypatch) -> None:
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(collection_name="collection"),
        ),
    )

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            raise QdrantException("qdrant timeout")

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.ChatService(runtime).answer(question="hello"))
    except DependencyUnavailableError as exc:
        assert exc.dependency == "qdrant"
        assert str(exc) == "Qdrant is unavailable"
    else:
        raise AssertionError("expected DependencyUnavailableError")


def test_module_answer_wraps_pymongo_errors_as_dependency_unavailable(monkeypatch) -> None:
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(collection_name="collection"),
        ),
    )

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            raise pymongo_errors.ServerSelectionTimeoutError("mongo down")

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.answer("hello", runtime))
    except DependencyUnavailableError as exc:
        assert exc.dependency == "mongodb"
        assert str(exc) == "MongoDB is unavailable"
    else:
        raise AssertionError("expected DependencyUnavailableError")


def test_chat_service_wraps_pymongo_errors_as_dependency_unavailable(monkeypatch) -> None:
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(collection_name="collection"),
        ),
    )

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            raise pymongo_errors.ServerSelectionTimeoutError("mongo down")

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.ChatService(runtime).answer(question="hello"))
    except DependencyUnavailableError as exc:
        assert exc.dependency == "mongodb"
        assert str(exc) == "MongoDB is unavailable"
    else:
        raise AssertionError("expected DependencyUnavailableError")


def test_module_answer_preserves_domain_errors(monkeypatch) -> None:
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(collection_name="collection"),
        ),
    )

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            raise NoContextRetrievedError("hello")

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.answer("hello", runtime))
    except NoContextRetrievedError as exc:
        assert exc.question == "hello"
    else:
        raise AssertionError("expected NoContextRetrievedError")


def test_chat_service_preserves_domain_errors(monkeypatch) -> None:
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(collection_name="collection"),
        ),
    )

    class FakeChatGraphService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, **kwargs):
            raise NoContextRetrievedError("hello")

    monkeypatch.setattr(chat_service, "ChatGraphService", FakeChatGraphService)

    try:
        asyncio.run(chat_service.ChatService(runtime).answer(question="hello"))
    except NoContextRetrievedError as exc:
        assert exc.question == "hello"
    else:
        raise AssertionError("expected NoContextRetrievedError")
