from __future__ import annotations

import asyncio
from types import SimpleNamespace

from rag_demo.answering import AnswerResult
from langchain_core.documents import Document

from services import chat_service


def test_chat_service_logs_total_duration(caplog, monkeypatch):
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(async_client="async-client", collection_name="collection"),
            mongo_repository=SimpleNamespace(),
        ),
        llm=object(),
        dense_embeddings=object(),
        sparse_embeddings=object(),
        reranker=object(),
    )
    expected = AnswerResult(
        answer="done",
        parent_chunks=[Document(page_content="context", metadata={"parent_id": "parent-1"})],
        source_items=[{"parent_id": "parent-1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    )

    async def fake_answer_query_async(**kwargs):
        return expected

    monkeypatch.setattr(chat_service, "answer_query_async", fake_answer_query_async)

    async def run() -> AnswerResult:
        return await chat_service.answer("hello", runtime)

    with caplog.at_level("INFO"):
        result = asyncio.run(run())

    assert result is expected
    completed = next(record for record in caplog.records if getattr(record, "event", None) == "chat.answer.completed")
    assert completed.question == "hello"
    assert completed.duration_ms >= 0


def test_chat_service_logs_failures(caplog, monkeypatch):
    runtime = SimpleNamespace(
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(async_client="async-client", collection_name="collection"),
            mongo_repository=SimpleNamespace(),
        ),
        llm=object(),
        dense_embeddings=object(),
        sparse_embeddings=object(),
        reranker=object(),
    )

    async def raise_error(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_service, "answer_query_async", raise_error)

    async def run() -> None:
        await chat_service.answer("hello", runtime)

    with caplog.at_level("INFO"):
        try:
            asyncio.run(run())
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

    failed = next(record for record in caplog.records if getattr(record, "event", None) == "chat.answer.failed")
    assert failed.question == "hello"
    assert failed.error == "boom"
