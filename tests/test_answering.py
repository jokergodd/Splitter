from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document

from rag_demo import answering


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class _FakeAsyncLLM(_FakeLLM):
    async def ainvoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


def test_build_answer_prompt_uses_original_query_and_parent_chunks_only():
    parent_chunks = [
        Document(page_content="父块一内容", metadata={"parent_id": "parent-1"}),
        Document(page_content="父块二内容", metadata={"parent_id": "parent-2"}),
    ]

    prompt = answering.build_answer_prompt("原始问题", parent_chunks)

    assert "原始问题" in prompt
    assert "父块一内容" in prompt
    assert "父块二内容" in prompt
    assert "rewrite" not in prompt.lower()


def test_answer_query_returns_structured_result(monkeypatch):
    llm = _FakeLLM("最终答案")
    calls: list[tuple[str, object]] = []

    rewrite_result = SimpleNamespace(
        original_query="原始问题",
        rewritten_queries=["原始问题", "候选改写一"],
    )

    candidate_hits = [
        answering.HybridRetrievalHit(
            child_id="child-1",
            score=0.2,
            payload={"parent_id": "parent-1", "child_id": "child-1", "text": "child-one"},
            point_id="point-1",
        ),
        answering.HybridRetrievalHit(
            child_id="child-2",
            score=0.8,
            payload={"parent_id": "parent-2", "child_id": "child-2", "text": "child-two"},
            point_id="point-2",
        ),
    ]
    reranked_documents = [
        Document(
            page_content="child-two",
            metadata={"parent_id": "parent-2", "child_id": "child-2", "rerank_score": 0.9},
        ),
        Document(
            page_content="child-one",
            metadata={"parent_id": "parent-1", "child_id": "child-1", "rerank_score": 0.3},
        ),
    ]
    collapsed_parent_hits = [
        Document(
            page_content="child-two",
            metadata={"parent_id": "parent-2", "child_id": "child-2", "rerank_score": 0.9},
        ),
        Document(
            page_content="child-one",
            metadata={"parent_id": "parent-1", "child_id": "child-1", "rerank_score": 0.3},
        ),
    ]
    parent_chunks = [
        Document(
            page_content="parent-two",
            metadata={
                "parent_id": "parent-2",
                "source": "docs/parent-two.md",
                "file_path": "/tmp/parent-two.md",
            },
        ),
        Document(
            page_content="parent-one",
            metadata={
                "parent_id": "parent-1",
                "source": "docs/parent-one.md",
                "file_path": "/tmp/parent-one.md",
            },
        ),
    ]

    monkeypatch.setattr(
        answering,
        "rewrite_queries",
        lambda original_query, llm, max_queries=4: calls.append(("rewrite_queries", original_query))
        or rewrite_result,
    )
    monkeypatch.setattr(
        answering,
        "query_hybrid_children_for_queries",
        lambda **kwargs: calls.append(
            ("query_hybrid_children_for_queries", list(kwargs["query_texts"]))
        )
        or candidate_hits,
    )
    monkeypatch.setattr(
        answering,
        "rerank_candidates",
        lambda original_query, candidates, reranker, limit=10: calls.append(
            ("rerank_candidates", original_query)
        )
        or reranked_documents,
    )
    monkeypatch.setattr(
        answering,
        "collapse_to_parent_hits",
        lambda reranked_candidates, limit=5: calls.append(
            ("collapse_to_parent_hits", [doc.metadata["parent_id"] for doc in reranked_candidates])
        )
        or collapsed_parent_hits,
    )
    monkeypatch.setattr(
        answering,
        "fetch_parent_chunks",
        lambda parent_ids, mongo_repository: calls.append(("fetch_parent_chunks", parent_ids))
        or parent_chunks,
    )
    monkeypatch.setattr(
        answering,
        "build_answer_prompt",
        lambda original_query, parent_chunks: calls.append(("build_answer_prompt", original_query))
        or f"PROMPT::{original_query}::{','.join(doc.page_content for doc in parent_chunks)}",
    )

    result = answering.answer_query(
        original_query="原始问题",
        llm=llm,
        client=SimpleNamespace(),
        collection_name="child_chunks_hybrid",
        embeddings=SimpleNamespace(),
        sparse_embeddings=SimpleNamespace(),
        mongo_repository=SimpleNamespace(),
    )

    assert result.answer == "最终答案"
    assert result.rewritten_queries == ["原始问题", "候选改写一"]
    assert [chunk.metadata["parent_id"] for chunk in result.parent_chunks] == ["parent-2", "parent-1"]
    assert result.source_items == [
        {
            "parent_id": "parent-2",
            "source": "docs/parent-two.md",
            "file_path": "/tmp/parent-two.md",
        },
        {
            "parent_id": "parent-1",
            "source": "docs/parent-one.md",
            "file_path": "/tmp/parent-one.md",
        },
    ]
    assert llm.calls == ["PROMPT::原始问题::parent-two,parent-one"]
    assert calls == [
        ("rewrite_queries", "原始问题"),
        ("query_hybrid_children_for_queries", ["原始问题", "候选改写一"]),
        ("rerank_candidates", "原始问题"),
        ("collapse_to_parent_hits", ["parent-2", "parent-1"]),
        ("fetch_parent_chunks", ["parent-2", "parent-1"]),
        ("build_answer_prompt", "原始问题"),
    ]


def test_answer_query_accepts_llm_message_objects(monkeypatch):
    llm = SimpleNamespace(invoke=lambda prompt: _FakeMessage("最终答案"))
    monkeypatch.setattr(
        answering,
        "rewrite_queries",
        lambda original_query, llm, max_queries=4: SimpleNamespace(rewritten_queries=[original_query]),
    )
    monkeypatch.setattr(answering, "query_hybrid_children_for_queries", lambda **kwargs: [])
    monkeypatch.setattr(answering, "rerank_candidates", lambda original_query, candidates, reranker, limit=10: [])
    monkeypatch.setattr(answering, "collapse_to_parent_hits", lambda reranked_candidates, limit=5: [])
    monkeypatch.setattr(answering, "fetch_parent_chunks", lambda parent_ids, mongo_repository: [])
    monkeypatch.setattr(answering, "build_answer_prompt", lambda original_query, parent_chunks: "PROMPT")

    result = answering.answer_query(
        original_query="原始问题",
        llm=llm,
        client=SimpleNamespace(),
        collection_name="child_chunks_hybrid",
        embeddings=SimpleNamespace(),
        sparse_embeddings=SimpleNamespace(),
        mongo_repository=SimpleNamespace(),
    )

    assert result.answer == "最终答案"


def test_answer_query_async_uses_async_rewrite_retrieval_and_llm(monkeypatch):
    llm = _FakeAsyncLLM("最终答案")
    calls: list[tuple[str, object]] = []

    rewrite_result = SimpleNamespace(
        original_query="原始问题",
        rewritten_queries=["原始问题", "候选改写一"],
    )
    candidate_hits = [
        answering.HybridRetrievalHit(
            child_id="child-2",
            score=0.8,
            payload={"parent_id": "parent-2", "child_id": "child-2", "text": "child-two"},
            point_id="point-2",
        )
    ]
    reranked_documents = [
        Document(
            page_content="child-two",
            metadata={"parent_id": "parent-2", "child_id": "child-2", "rerank_score": 0.9},
        )
    ]
    parent_chunks = [
        Document(
            page_content="parent-two",
            metadata={
                "parent_id": "parent-2",
                "source": "docs/parent-two.md",
                "file_path": "/tmp/parent-two.md",
            },
        )
    ]

    async def fake_rewrite_queries_async(original_query, llm, max_queries=4):
        calls.append(("rewrite_queries_async", original_query))
        return rewrite_result

    async def fake_query_hybrid_children_for_queries_async(**kwargs):
        calls.append(("query_hybrid_children_for_queries_async", list(kwargs["query_texts"])))
        return candidate_hits

    async def fake_fetch_parent_chunks_async(parent_ids, mongo_repository):
        calls.append(("fetch_parent_chunks_async", parent_ids))
        return parent_chunks

    def fake_rerank_candidates(original_query, candidates, reranker, limit=10):
        calls.append(("rerank_candidates", original_query))
        return reranked_documents

    monkeypatch.setattr(answering, "rewrite_queries_async", fake_rewrite_queries_async)
    monkeypatch.setattr(
        answering,
        "query_hybrid_children_for_queries_async",
        fake_query_hybrid_children_for_queries_async,
    )
    monkeypatch.setattr(answering, "fetch_parent_chunks_async", fake_fetch_parent_chunks_async)
    monkeypatch.setattr(answering, "rerank_candidates", fake_rerank_candidates)
    monkeypatch.setattr(
        answering,
        "build_answer_prompt",
        lambda original_query, parent_chunks: calls.append(("build_answer_prompt", original_query))
        or "PROMPT",
    )

    result = asyncio.run(
        answering.answer_query_async(
            original_query="原始问题",
            llm=llm,
            client=SimpleNamespace(),
            collection_name="child_chunks_hybrid",
            embeddings=SimpleNamespace(),
            sparse_embeddings=SimpleNamespace(),
            mongo_repository=SimpleNamespace(),
            reranker=SimpleNamespace(),
        )
    )

    assert result.answer == "最终答案"
    assert result.rewritten_queries == ["原始问题", "候选改写一"]
    assert llm.calls == ["PROMPT"]
    assert calls == [
        ("rewrite_queries_async", "原始问题"),
        ("query_hybrid_children_for_queries_async", ["原始问题", "候选改写一"]),
        ("rerank_candidates", "原始问题"),
        ("fetch_parent_chunks_async", ["parent-2"]),
        ("build_answer_prompt", "原始问题"),
    ]
