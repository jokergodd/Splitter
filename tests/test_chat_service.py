from __future__ import annotations

import asyncio
from types import SimpleNamespace

from rag_demo.answering import AnswerResult
from langchain_core.documents import Document
from qdrant_client.http.exceptions import UnexpectedResponse

from services import chat_service
from services.errors import CollectionNotReadyError, NoContextRetrievedError


def test_answer_uses_async_answering_and_forwards_runtime_dependencies(monkeypatch):
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
    expected = AnswerResult(
        answer="done",
        parent_chunks=[Document(page_content="context", metadata={"parent_id": "parent-1"})],
        source_items=[{"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    )
    captured: dict[str, object] = {}

    async def fake_answer_query_async(**kwargs):
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(chat_service, "answer_query_async", fake_answer_query_async)

    async def run() -> AnswerResult:
        return await chat_service.answer("hello", runtime, top_k=3, candidate_limit=7)

    result = asyncio.run(run())

    assert result is expected
    assert captured["kwargs"]["original_query"] == "hello"
    assert captured["kwargs"]["llm"] is runtime.llm
    assert captured["kwargs"]["client"] == "async-client"
    assert captured["kwargs"]["collection_name"] == "collection"
    assert captured["kwargs"]["embeddings"] is runtime.dense_embeddings
    assert captured["kwargs"]["sparse_embeddings"] is runtime.sparse_embeddings
    assert captured["kwargs"]["mongo_repository"] == "mongo"
    assert captured["kwargs"]["reranker"] is runtime.reranker
    assert captured["kwargs"]["top_k"] == 3
    assert captured["kwargs"]["candidate_limit"] == 7


def test_answer_propagates_sync_errors(monkeypatch):
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

    async def fake_answer_query_async(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_service, "answer_query_async", fake_answer_query_async)

    async def run() -> None:
        await chat_service.answer("hello", runtime)

    try:
        asyncio.run(run())
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")


def test_answer_wraps_missing_collection_as_collection_not_ready(monkeypatch):
    runtime = SimpleNamespace(
        llm=object(),
        dense_embeddings=object(),
        sparse_embeddings=object(),
        reranker=object(),
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(async_client="async-client", collection_name="child_chunks_hybrid"),
            mongo_repository="mongo",
        ),
    )

    async def fake_answer_query_async(**kwargs):
        raise UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"{\"status\":{\"error\":\"Not found: Collection `child_chunks_hybrid` doesn't exist!\"}}",
            headers={},
        )

    monkeypatch.setattr(chat_service, "answer_query_async", fake_answer_query_async)

    async def run() -> None:
        await chat_service.answer("hello", runtime)

    try:
        asyncio.run(run())
    except CollectionNotReadyError as exc:
        assert exc.collection_name == "child_chunks_hybrid"
    else:
        raise AssertionError("expected CollectionNotReadyError")


def test_answer_raises_no_context_retrieved_when_parent_chunks_are_empty(monkeypatch):
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

    async def fake_answer_query_async(**kwargs):
        return AnswerResult(answer="done", parent_chunks=[], source_items=[])

    monkeypatch.setattr(chat_service, "answer_query_async", fake_answer_query_async)

    async def run() -> None:
        await chat_service.answer("hello", runtime)

    try:
        asyncio.run(run())
    except NoContextRetrievedError as exc:
        assert exc.question == "hello"
    else:
        raise AssertionError("expected NoContextRetrievedError")


def test_chat_service_class_delegates_to_module_answer(monkeypatch):
    runtime = SimpleNamespace()
    expected = AnswerResult(answer="delegated")
    captured: dict[str, object] = {}

    async def fake_answer(question, runtime_arg, **kwargs):
        captured["question"] = question
        captured["runtime"] = runtime_arg
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(chat_service, "answer", fake_answer)

    async def run() -> AnswerResult:
        service = chat_service.ChatService(runtime)
        return await service.answer(question="hello", top_k=2)

    result = asyncio.run(run())

    assert result is expected
    assert captured == {
        "question": "hello",
        "runtime": runtime,
        "kwargs": {
            "top_k": 2,
            "candidate_limit": 30,
            "max_queries": 4,
            "parent_limit": 5,
        },
    }
