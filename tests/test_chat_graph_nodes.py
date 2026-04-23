from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from graphs.chat.models import ChatGraphInput
from graphs.chat.nodes import build_nodes
from graphs.chat.state import initialize_state
from rag_demo import answering
from rag_demo.retrieval import HybridRetrievalHit


def test_rewrite_node_falls_back_to_original_query_on_error():
    class FakeDeps:
        async def rewrite_queries(self, *, question: str, max_queries: int):
            raise RuntimeError("rewrite down")

    nodes = build_nodes(FakeDeps())
    state = initialize_state(ChatGraphInput(question="原问题", request_id="req-1"))

    updated = asyncio.run(nodes.rewrite_query(state))

    assert updated["rewritten_queries"] == ["原问题"]
    assert updated["rewrite_status"] == "fallback"
    assert updated["errors"] == [{"stage": "rewrite_query", "error": "rewrite down"}]


def test_rerank_fallback_keeps_parent_recall_order_stable():
    class FakeDeps:
        reranker = object()

    nodes = build_nodes(FakeDeps())
    state = initialize_state(
        ChatGraphInput(question="原问题", request_id="req-1", candidate_limit=3, parent_limit=2)
    )
    state["merged_children"] = [
        Document(page_content="child-p1", metadata={"parent_id": "p1", "child_id": "c1", "retrieval_score": 0.2}),
        Document(page_content="child-p2", metadata={"parent_id": "p2", "child_id": "c2", "retrieval_score": 0.9}),
        Document(page_content="child-p3", metadata={"parent_id": "p3", "child_id": "c3", "retrieval_score": 0.6}),
    ]

    updated = asyncio.run(nodes.rerank_candidates(state))

    assert [doc.metadata["parent_id"] for doc in updated["reranked_children"]] == ["p2", "p3", "p1"]
    assert [doc.metadata["rerank_score"] for doc in updated["reranked_children"]] == [0.9, 0.6, 0.2]
    assert updated["rerank_status"] == "fallback"
    assert updated["errors"][0]["stage"] == "rerank_candidates"
    assert updated["errors"][0]["error"]


def test_retrieve_and_merge_candidates_reuse_answering_helper(monkeypatch):
    class FakeDeps:
        qdrant_client = object()
        collection_name = "collection"
        dense_embeddings = object()
        sparse_embeddings = object()

    captured: dict[str, object] = {}
    merged_documents = [
        Document(page_content="merged", metadata={"parent_id": "p2", "child_id": "c2", "retrieval_score": 0.9})
    ]

    async def fake_retrieve_candidate_documents_async(**kwargs):
        captured["kwargs"] = kwargs
        return merged_documents

    monkeypatch.setattr(answering, "retrieve_candidate_documents_async", fake_retrieve_candidate_documents_async)

    nodes = build_nodes(FakeDeps())
    state = initialize_state(
        ChatGraphInput(question="原问题", request_id="req-1", top_k=4, candidate_limit=7)
    )
    state["rewritten_queries"] = ["原问题", "候选问题"]

    retrieved = asyncio.run(nodes.retrieve_candidates(state))
    merged = nodes.merge_candidates(retrieved)

    assert captured["kwargs"] == {
        "client": FakeDeps.qdrant_client,
        "collection_name": "collection",
        "query_texts": ["原问题", "候选问题"],
        "embeddings": FakeDeps.dense_embeddings,
        "sparse_embeddings": FakeDeps.sparse_embeddings,
        "top_k": 4,
        "candidate_limit": 7,
    }
    assert retrieved["retrieved_children"] == merged_documents
    assert merged["merged_children"] == merged_documents


def test_answering_candidate_document_helpers_match_retrieval_merge_behavior():
    hit_groups = [
        [
            HybridRetrievalHit(
                child_id="c1",
                score=0.2,
                payload={"parent_id": "p1", "child_id": "c1", "text": "doc-1"},
                point_id="point-1",
            ),
            HybridRetrievalHit(
                child_id="c2",
                score=0.4,
                payload={"parent_id": "p2", "child_id": "c2", "text": "doc-2a"},
                point_id="point-2a",
            ),
        ],
        [
            HybridRetrievalHit(
                child_id="c2",
                score=0.9,
                payload={"parent_id": "p2", "child_id": "c2", "text": "doc-2b"},
                point_id="point-2b",
            ),
            HybridRetrievalHit(
                child_id="c3",
                score=0.5,
                payload={"parent_id": "p3", "child_id": "c3", "text": "doc-3"},
                point_id="point-3",
            ),
        ],
    ]

    merged = answering.merge_retrieved_child_hit_groups(hit_groups, candidate_limit=3)

    assert [doc.metadata["child_id"] for doc in merged] == ["c2", "c3", "c1"]
    assert [doc.metadata["retrieval_score"] for doc in merged] == [0.9, 0.5, 0.2]
    assert [doc.page_content for doc in merged] == ["doc-2b", "doc-3", "doc-1"]


def test_build_response_node_uses_parent_chunk_sources():
    class FakeDeps:
        pass

    nodes = build_nodes(FakeDeps())
    state = initialize_state(ChatGraphInput(question="原问题", request_id="req-1"))
    state["answer"] = "答案"
    state["parent_chunks"] = [
        Document(page_content="ctx", metadata={"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"})
    ]
    state["source_items"] = [{"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"}]

    updated = nodes.build_response(state)

    assert updated["response_payload"] == {
        "answer": "答案",
        "source_items": [{"parent_id": "p1", "source": "doc.md", "file_path": "/tmp/doc.md"}],
    }
