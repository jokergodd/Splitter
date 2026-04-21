from __future__ import annotations

from langchain_core.documents import Document

import rag_demo.rerank as rerank
import rag_demo.reranker_runtime as reranker_runtime


class _FakeCrossEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        return [float(len(query) + len(document)) for query, document in pairs]


def test_default_reranker_model_matches_expected_name():
    assert reranker_runtime.DEFAULT_RERANKER_MODEL == "BAAI/bge-reranker-base"


def test_cross_encoder_scorer_scores_documents_in_batch(monkeypatch):
    fake_encoder = _FakeCrossEncoder("unused")
    monkeypatch.setattr(reranker_runtime, "CrossEncoder", lambda model_name: fake_encoder)

    scorer = reranker_runtime.build_cross_encoder_reranker()
    scores = scorer.score_batch("query", ["alpha", "beta"])

    assert scorer.model_name == reranker_runtime.DEFAULT_RERANKER_MODEL
    assert fake_encoder.model_name == "unused"
    assert fake_encoder.calls == [[("query", "alpha"), ("query", "beta")]]
    assert scores == [10.0, 9.0]


def test_cross_encoder_scorer_is_compatible_with_rerank_candidates(monkeypatch):
    fake_encoder = _FakeCrossEncoder("unused")
    monkeypatch.setattr(reranker_runtime, "CrossEncoder", lambda model_name: fake_encoder)

    scorer = reranker_runtime.build_cross_encoder_reranker()
    candidates = [
        Document(page_content="a", metadata={"child_id": "child-1"}),
        Document(page_content="bb", metadata={"child_id": "child-2"}),
        Document(page_content="ccc", metadata={"child_id": "child-3"}),
    ]

    reranked = rerank.rerank_candidates("query", candidates, scorer, limit=2)

    assert [candidate.page_content for candidate in reranked] == ["ccc", "bb"]
    assert [candidate.metadata["rerank_score"] for candidate in reranked] == [8.0, 7.0]
