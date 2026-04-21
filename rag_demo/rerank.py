from __future__ import annotations

from langchain_core.documents import Document


def _score_candidate(original_query: str, candidate: Document, reranker) -> float:
    if callable(reranker):
        return float(reranker(original_query, candidate))
    if hasattr(reranker, "score"):
        return float(reranker.score(original_query, candidate))
    raise TypeError("reranker must be callable or expose a score method")


def rerank_candidates(original_query: str, candidates: list[Document], reranker, limit: int = 10) -> list[Document]:
    scored_candidates: list[tuple[float, int, Document]] = []

    for index, candidate in enumerate(candidates):
        score = _score_candidate(original_query, candidate, reranker)
        metadata = dict(candidate.metadata)
        metadata["rerank_score"] = score
        scored_candidates.append(
            (
                score,
                index,
                Document(page_content=candidate.page_content, metadata=metadata, id=candidate.id),
            )
        )

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored_candidates[:limit]]


__all__ = ["rerank_candidates"]
