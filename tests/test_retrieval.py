from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from qdrant_client.http.models.models import Fusion, FusionQuery, Prefetch, SparseVector

from rag_demo import retrieval


class _FakeEmbeddings:
    def __init__(self):
        self.query_calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [float(len(text)), 0.25]


class _FakeSparseEmbeddings:
    def __init__(self):
        self.query_calls: list[str] = []

    def embed_query(self, text: str):
        self.query_calls.append(text)
        return {"indices": [1, 3], "values": [0.5, float(len(text))]}


class _FakeSparseQueryVector:
    def __init__(self, indices: list[int], values: list[float]):
        self.indices = indices
        self.values = values

    def as_object(self):
        return {"indices": self.indices, "values": self.values}


class _FastEmbedStyleSparseEmbeddings:
    def __init__(self):
        self.query_calls: list[tuple[str, ...]] = []

    def query_embed(self, texts: list[str]):
        self.query_calls.append(tuple(texts))
        return [_FakeSparseQueryVector(indices=[2, 4], values=[0.7, 1.1])]


class _FakeQdrantClient:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_query_hybrid_children_uses_dense_and_sparse_prefetch_with_rrf():
    response = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="point-1",
                score=0.9,
                payload={"child_id": "child-1", "text": "alpha"},
            )
        ]
    )
    client = _FakeQdrantClient(response=response)

    hits = retrieval.query_hybrid_children(
        client=client,
        collection_name="child_chunks_hybrid",
        query_text="hello",
        embeddings=_FakeEmbeddings(),
        sparse_embeddings=_FakeSparseEmbeddings(),
        top_k=3,
        candidate_limit=11,
    )

    assert len(hits) == 1
    assert hits[0].child_id == "child-1"
    assert hits[0].score == 0.9

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["collection_name"] == "child_chunks_hybrid"
    assert call["query"] == FusionQuery(fusion=Fusion.RRF)
    assert call["limit"] == 3
    assert call["with_payload"] is True
    assert call["prefetch"] == [
        Prefetch(query=[5.0, 0.25], using="dense", limit=11),
        Prefetch(query=SparseVector(indices=[1, 3], values=[0.5, 5.0]), using="sparse", limit=11),
    ]


def test_query_hybrid_children_for_queries_dedupes_by_child_id_and_caps_candidate_pool():
    first = [
        retrieval.HybridRetrievalHit(child_id="child-1", score=0.3, payload={"child_id": "child-1"}),
        retrieval.HybridRetrievalHit(child_id="child-2", score=0.6, payload={"child_id": "child-2"}),
    ]
    second = [
        retrieval.HybridRetrievalHit(child_id="child-1", score=0.9, payload={"child_id": "child-1"}),
        retrieval.HybridRetrievalHit(child_id="child-3", score=0.8, payload={"child_id": "child-3"}),
        retrieval.HybridRetrievalHit(child_id="child-4", score=0.7, payload={"child_id": "child-4"}),
    ]

    calls: list[str] = []

    def fake_single_query(*, query_text: str, **kwargs):
        calls.append(query_text)
        return first if query_text == "q1" else second

    merged = retrieval.query_hybrid_children_for_queries(
        client=SimpleNamespace(),
        collection_name="child_chunks_hybrid",
        query_texts=["q1", "q2"],
        embeddings=_FakeEmbeddings(),
        sparse_embeddings=_FakeSparseEmbeddings(),
        top_k=5,
        candidate_limit=2,
        single_query_fn=fake_single_query,
    )

    assert calls == ["q1", "q2"]
    assert [hit.child_id for hit in merged] == ["child-1", "child-3"]
    assert [hit.score for hit in merged] == [0.9, 0.8]


def test_query_hybrid_children_supports_fastembed_style_query_embed():
    response = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="point-1",
                score=0.9,
                payload={"child_id": "child-1", "text": "alpha"},
            )
        ]
    )
    client = _FakeQdrantClient(response=response)
    sparse_embeddings = _FastEmbedStyleSparseEmbeddings()

    hits = retrieval.query_hybrid_children(
        client=client,
        collection_name="child_chunks_hybrid",
        query_text="hello",
        embeddings=_FakeEmbeddings(),
        sparse_embeddings=sparse_embeddings,
    )

    assert len(hits) == 1
    assert sparse_embeddings.query_calls == [("hello",)]
    assert client.calls[0]["prefetch"][1] == Prefetch(
        query=SparseVector(indices=[2, 4], values=[0.7, 1.1]),
        using="sparse",
        limit=10,
    )


def test_merge_hybrid_hits_keeps_highest_score_per_child_id():
    merged = retrieval.merge_hybrid_hits(
        [
            retrieval.HybridRetrievalHit(child_id="child-1", score=0.2, payload={"child_id": "child-1"}),
            retrieval.HybridRetrievalHit(child_id="child-2", score=0.8, payload={"child_id": "child-2"}),
        ],
        [
            retrieval.HybridRetrievalHit(child_id="child-1", score=0.6, payload={"child_id": "child-1"}),
            retrieval.HybridRetrievalHit(child_id="child-3", score=0.4, payload={"child_id": "child-3"}),
        ],
        candidate_limit=10,
    )

    assert [hit.child_id for hit in merged] == ["child-2", "child-1", "child-3"]
    assert [hit.score for hit in merged] == [0.8, 0.6, 0.4]
