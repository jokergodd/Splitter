from __future__ import annotations

from langchain_core.documents import Document

import rag_demo.rerank as rerank


class _RecordingReranker:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls: list[tuple[str, str]] = []

    def __call__(self, query: str, candidate: Document) -> float:
        self.calls.append((query, candidate.page_content))
        return self.scores[candidate.page_content]


def test_rerank_candidates_scores_only_with_original_query_and_returns_top_candidates():
    candidates = [
        Document(page_content="alpha", metadata={"parent_id": "parent-1", "child_id": "child-1"}),
        Document(page_content="beta", metadata={"parent_id": "parent-2", "child_id": "child-2"}),
        Document(page_content="gamma", metadata={"parent_id": "parent-3", "child_id": "child-3"}),
    ]
    reranker = _RecordingReranker(
        {
            "alpha": 0.4,
            "beta": 0.9,
            "gamma": 0.7,
        }
    )

    reranked = rerank.rerank_candidates("original query", candidates, reranker, limit=2)

    assert reranker.calls == [
        ("original query", "alpha"),
        ("original query", "beta"),
        ("original query", "gamma"),
    ]
    assert [candidate.page_content for candidate in reranked] == ["beta", "gamma"]
    assert [candidate.metadata["rerank_score"] for candidate in reranked] == [0.9, 0.7]
