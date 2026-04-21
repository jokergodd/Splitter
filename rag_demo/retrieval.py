from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from qdrant_client.http.models.models import Fusion, FusionQuery, Prefetch, SparseVector


@dataclass(slots=True)
class HybridRetrievalHit:
    child_id: str
    score: float
    payload: dict[str, Any]
    point_id: str | None = None


def _coerce_sparse_vector(vector: Any) -> SparseVector:
    if isinstance(vector, SparseVector):
        return vector
    if isinstance(vector, dict):
        return SparseVector(indices=list(vector["indices"]), values=list(vector["values"]))
    raise TypeError("sparse query vectors must be a SparseVector or a mapping with indices/values")


def _embed_sparse_query(sparse_embeddings, query_text: str) -> SparseVector:
    if hasattr(sparse_embeddings, "embed_query"):
        return _coerce_sparse_vector(sparse_embeddings.embed_query(query_text))
    if hasattr(sparse_embeddings, "query_embed"):
        vectors = list(sparse_embeddings.query_embed([query_text]))
        if len(vectors) != 1:
            raise ValueError("sparse query embedding must return exactly one vector for one query")
        return _coerce_sparse_vector(vectors[0].as_object())
    raise TypeError("sparse_embeddings must expose embed_query(...) or query_embed(...)")


def _extract_hybrid_hit(point: Any) -> HybridRetrievalHit:
    payload = dict(getattr(point, "payload", None) or {})
    child_id = payload.get("child_id")
    if child_id is None:
        raise ValueError("hybrid retrieval points must include child_id in the payload")

    return HybridRetrievalHit(
        child_id=str(child_id),
        score=float(getattr(point, "score", 0.0)),
        payload=payload,
        point_id=None if getattr(point, "id", None) is None else str(point.id),
    )


def query_hybrid_children(
    *,
    client,
    collection_name: str,
    query_text: str,
    embeddings,
    sparse_embeddings,
    top_k: int = 10,
    candidate_limit: int | None = None,
    dense_vector_name: str = "dense",
    sparse_vector_name: str = "sparse",
    with_payload: bool = True,
) -> list[HybridRetrievalHit]:
    dense_vector = embeddings.embed_query(query_text)
    sparse_vector = _embed_sparse_query(sparse_embeddings, query_text)
    prefetch_limit = candidate_limit or top_k

    response = client.query_points(
        collection_name=collection_name,
        query=FusionQuery(fusion=Fusion.RRF),
        prefetch=[
            Prefetch(query=dense_vector, using=dense_vector_name, limit=prefetch_limit),
            Prefetch(query=sparse_vector, using=sparse_vector_name, limit=prefetch_limit),
        ],
        limit=top_k,
        with_payload=with_payload,
    )

    return [_extract_hybrid_hit(point) for point in getattr(response, "points", response)]


def merge_hybrid_hits(
    *hit_groups: Sequence[HybridRetrievalHit],
    candidate_limit: int,
) -> list[HybridRetrievalHit]:
    best_hits: dict[str, tuple[int, HybridRetrievalHit]] = {}
    insertion_index = 0

    for group in hit_groups:
        for hit in group:
            current = best_hits.get(hit.child_id)
            if current is None or hit.score > current[1].score:
                best_hits[hit.child_id] = (insertion_index, hit)
            insertion_index += 1

    merged_hits = sorted(
        best_hits.values(),
        key=lambda item: (-item[1].score, item[0]),
    )
    return [hit for _, hit in merged_hits[:candidate_limit]]


def query_hybrid_children_for_queries(
    *,
    client,
    collection_name: str,
    query_texts: Iterable[str],
    embeddings,
    sparse_embeddings,
    top_k: int = 10,
    candidate_limit: int = 30,
    dense_vector_name: str = "dense",
    sparse_vector_name: str = "sparse",
    single_query_fn: Callable[..., list[HybridRetrievalHit]] | None = None,
) -> list[HybridRetrievalHit]:
    single_query_fn = single_query_fn or query_hybrid_children
    hit_groups: list[list[HybridRetrievalHit]] = []

    for query_text in query_texts:
        hit_groups.append(
            single_query_fn(
                client=client,
                collection_name=collection_name,
                query_text=query_text,
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
                top_k=top_k,
                candidate_limit=candidate_limit,
                dense_vector_name=dense_vector_name,
                sparse_vector_name=sparse_vector_name,
            )
        )

    return merge_hybrid_hits(*hit_groups, candidate_limit=candidate_limit)


__all__ = [
    "HybridRetrievalHit",
    "merge_hybrid_hits",
    "query_hybrid_children",
    "query_hybrid_children_for_queries",
]
