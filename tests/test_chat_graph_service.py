from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document

from rag_demo.answering import AnswerResult
import services.chat_graph_service as chat_graph_service_module
from services.chat_graph_service import ChatGraphService


def test_chat_graph_service_returns_answer_result() -> None:
    class FakeGraph:
        async def ainvoke(self, payload: dict):
            assert payload == {
                "question": "hello",
                "top_k": 10,
                "candidate_limit": 30,
                "max_queries": 4,
                "parent_limit": 5,
                "request_id": "req-123",
            }
            return {
                "rewritten_queries": ["hello", "hello rewritten"],
                "parent_chunks": [
                    Document(
                        page_content="context",
                        metadata={"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"},
                    )
                ],
                "response_payload": {
                    "answer": "graph:hello",
                    "source_items": [{"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
                }
            }

    service = ChatGraphService(runtime=SimpleNamespace(), graph_app=FakeGraph())

    result = asyncio.run(service.answer(question="hello", request_id="req-123"))

    assert result == AnswerResult(
        answer="graph:hello",
        rewritten_queries=["hello", "hello rewritten"],
        parent_chunks=[
            Document(
                page_content="context",
                metadata={"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"},
            )
        ],
        source_items=[{"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    )


def test_chat_graph_service_reuses_graph_app_for_same_runtime(monkeypatch) -> None:
    build_calls: list[object] = []

    class FakeGraph:
        async def ainvoke(self, payload: dict):
            return {"response_payload": {"answer": payload["question"], "source_items": []}}

    def fake_build_chat_graph(deps):
        build_calls.append(deps)
        return FakeGraph()

    runtime = SimpleNamespace(
        llm=object(),
        dense_embeddings=object(),
        sparse_embeddings=object(),
        reranker=object(),
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(async_client="client", collection_name="collection"),
            mongo_repository="mongo",
        ),
    )

    monkeypatch.setattr(chat_graph_service_module, "build_chat_graph", fake_build_chat_graph)

    first_service = ChatGraphService(runtime)
    second_service = ChatGraphService(runtime)

    first_result = asyncio.run(first_service.answer(question="first"))
    second_result = asyncio.run(second_service.answer(question="second"))

    assert first_result == AnswerResult(answer="first", source_items=[])
    assert second_result == AnswerResult(answer="second", source_items=[])
    assert len(build_calls) == 1


def test_chat_graph_service_bubbles_llm_failure_from_generate_answer(monkeypatch) -> None:
    class FailingLlm:
        def invoke(self, prompt: str):
            raise RuntimeError("llm down")

    async def fake_rewrite_queries_async(question: str, llm, max_queries: int):
        return SimpleNamespace(rewritten_queries=[question])

    async def fake_retrieve_candidate_documents_async(**kwargs):
        return [
            Document(
                page_content="child context",
                metadata={"parent_id": "p1", "child_id": "c1", "retrieval_score": 0.9},
            )
        ]

    def fake_rerank_candidates(question: str, candidates: list[Document], reranker, limit: int):
        return candidates

    def fake_collapse_to_parent_hits(documents: list[Document], limit: int):
        return documents

    async def fake_fetch_parent_chunks_async(parent_ids: list[str], mongo_repository):
        return [
            Document(
                page_content="parent context",
                metadata={"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"},
            )
        ]

    monkeypatch.setattr("graphs.chat.nodes.rewrite_queries_async", fake_rewrite_queries_async)
    monkeypatch.setattr("graphs.chat.nodes.answering.retrieve_candidate_documents_async", fake_retrieve_candidate_documents_async)
    monkeypatch.setattr("graphs.chat.nodes.rerank_candidates", fake_rerank_candidates)
    monkeypatch.setattr("graphs.chat.nodes.collapse_to_parent_hits", fake_collapse_to_parent_hits)
    monkeypatch.setattr("graphs.chat.nodes.fetch_parent_chunks_async", fake_fetch_parent_chunks_async)

    runtime = SimpleNamespace(
        llm=FailingLlm(),
        dense_embeddings=object(),
        sparse_embeddings=object(),
        reranker=None,
        storage_backend=SimpleNamespace(
            qdrant_store=SimpleNamespace(async_client="client", collection_name="collection"),
            mongo_repository="mongo",
        ),
    )

    try:
        asyncio.run(ChatGraphService(runtime).answer(question="hello"))
    except RuntimeError as exc:
        assert str(exc) == "llm down"
    else:
        raise AssertionError("expected RuntimeError")
