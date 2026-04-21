from __future__ import annotations

from langchain_core.documents import Document

import rag_demo.parent_recall as parent_recall


class _FakeParentCollection:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.calls: list[dict] = []

    def find(self, query: dict):
        self.calls.append(query)
        requested_ids = query["parent_id"]["$in"]
        shuffled = [doc for doc in self.documents if doc["parent_id"] in requested_ids]
        return list(reversed(shuffled))


class _FakeMongoRepository:
    def __init__(self, documents: list[dict]):
        self._parent_chunks = _FakeParentCollection(documents)


def test_collapse_to_parent_hits_keeps_best_child_per_parent():
    reranked_candidates = [
        Document(
            page_content="child-1",
            metadata={"parent_id": "parent-1", "child_id": "child-1", "rerank_score": 0.4},
        ),
        Document(
            page_content="child-2",
            metadata={"parent_id": "parent-2", "child_id": "child-2", "rerank_score": 0.9},
        ),
        Document(
            page_content="child-3",
            metadata={"parent_id": "parent-1", "child_id": "child-3", "rerank_score": 0.8},
        ),
    ]

    collapsed = parent_recall.collapse_to_parent_hits(reranked_candidates, limit=5)

    assert [candidate.metadata["parent_id"] for candidate in collapsed] == ["parent-2", "parent-1"]
    assert [candidate.metadata["child_id"] for candidate in collapsed] == ["child-2", "child-3"]
    assert [candidate.metadata["rerank_score"] for candidate in collapsed] == [0.9, 0.8]


def test_fetch_parent_chunks_restores_input_parent_order():
    repository = _FakeMongoRepository(
        [
            {"parent_id": "parent-1", "text": "first", "metadata": {"parent_id": "parent-1"}},
            {"parent_id": "parent-2", "text": "second", "metadata": {"parent_id": "parent-2"}},
            {"parent_id": "parent-3", "text": "third", "metadata": {"parent_id": "parent-3"}},
        ]
    )

    chunks = parent_recall.fetch_parent_chunks(["parent-3", "parent-1"], repository)

    assert [chunk.metadata["parent_id"] for chunk in chunks] == ["parent-3", "parent-1"]
    assert [chunk.page_content for chunk in chunks] == ["third", "first"]
