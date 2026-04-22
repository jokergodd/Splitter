from __future__ import annotations

from typing import Any, Sequence

from langchain_core.documents import Document

from evals.models import EvalSample, ExperimentConfig, RetrievalCheckpoint, StageTrace
from rag_demo.answering import build_answer_prompt
from rag_demo.parent_recall import collapse_to_parent_hits, fetch_parent_chunks
from rag_demo.query_rewrite import rewrite_queries
from rag_demo.rerank import rerank_candidates
from rag_demo.retrieval import HybridRetrievalHit, merge_hybrid_hits, query_hybrid_children


def _child_hit_to_document(hit: HybridRetrievalHit) -> Document:
    metadata = dict(hit.payload)
    metadata.setdefault("child_id", hit.child_id)
    metadata.setdefault("retrieval_score", hit.score)
    return Document(
        page_content=str(hit.payload.get("text", "")),
        metadata=metadata,
        id=hit.point_id,
    )


def _hits_checkpoint(stage_name: str, hits: Sequence[HybridRetrievalHit]) -> RetrievalCheckpoint:
    return RetrievalCheckpoint(
        stage_name=stage_name,
        child_ids=[hit.child_id for hit in hits],
        parent_ids=[
            str(hit.payload["parent_id"])
            for hit in hits
            if hit.payload.get("parent_id") is not None
        ],
        contexts=[str(hit.payload.get("text", "")) for hit in hits],
    )


def _hybrid_per_query_checkpoint(query_text: str, hits: Sequence[HybridRetrievalHit]) -> RetrievalCheckpoint:
    return RetrievalCheckpoint(
        stage_name="hybrid_per_query",
        query_text=query_text,
        child_ids=[hit.child_id for hit in hits],
        parent_ids=[
            str(hit.payload["parent_id"])
            for hit in hits
            if hit.payload.get("parent_id") is not None
        ],
        contexts=[str(hit.payload.get("text", "")) for hit in hits],
        items=[
            {
                "rank": index,
                "child_id": hit.child_id,
                "parent_id": None
                if hit.payload.get("parent_id") is None
                else str(hit.payload.get("parent_id")),
                "score": hit.score,
                "point_id": hit.point_id,
                "text": str(hit.payload.get("text", "")),
            }
            for index, hit in enumerate(hits, start=1)
        ],
    )


def _merged_candidates_checkpoint(
    *,
    query_hits: Sequence[tuple[str, Sequence[HybridRetrievalHit]]],
    merged_hits: Sequence[HybridRetrievalHit],
) -> RetrievalCheckpoint:
    provenance_by_child: dict[str, list[dict[str, Any]]] = {}
    for query_text, hits in query_hits:
        for rank, hit in enumerate(hits, start=1):
            provenance_by_child.setdefault(hit.child_id, []).append(
                {"query_text": query_text, "rank": rank, "score": hit.score}
            )

    return RetrievalCheckpoint(
        stage_name="merged_candidates",
        child_ids=[hit.child_id for hit in merged_hits],
        parent_ids=[
            str(hit.payload["parent_id"])
            for hit in merged_hits
            if hit.payload.get("parent_id") is not None
        ],
        contexts=[str(hit.payload.get("text", "")) for hit in merged_hits],
        items=[
            {
                "rank": index,
                "child_id": hit.child_id,
                "parent_id": None
                if hit.payload.get("parent_id") is None
                else str(hit.payload.get("parent_id")),
                "score": hit.score,
                "point_id": hit.point_id,
                "text": str(hit.payload.get("text", "")),
                "provenance": provenance_by_child.get(hit.child_id, []),
            }
            for index, hit in enumerate(merged_hits, start=1)
        ],
    )


def _documents_checkpoint(stage_name: str, documents: Sequence[Document]) -> RetrievalCheckpoint:
    return RetrievalCheckpoint(
        stage_name=stage_name,
        child_ids=[
            str(document.metadata["child_id"])
            for document in documents
            if document.metadata.get("child_id") is not None
        ],
        parent_ids=[
            str(document.metadata["parent_id"])
            for document in documents
            if document.metadata.get("parent_id") is not None
        ],
        contexts=[document.page_content for document in documents],
        items=[
            {
                "rank": index,
                "child_id": None
                if document.metadata.get("child_id") is None
                else str(document.metadata.get("child_id")),
                "parent_id": None
                if document.metadata.get("parent_id") is None
                else str(document.metadata.get("parent_id")),
                "rerank_score": document.metadata.get("rerank_score"),
                "retrieval_score": document.metadata.get("retrieval_score"),
                "point_id": document.id,
                "text": document.page_content,
            }
            for index, document in enumerate(documents, start=1)
        ],
    )


def _collapsed_parents_checkpoint(
    collapsed_parents: Sequence[Document],
    parent_chunks: Sequence[Document],
) -> RetrievalCheckpoint:
    parent_text_by_id = {
        str(parent_chunk.metadata["parent_id"]): parent_chunk.page_content
        for parent_chunk in parent_chunks
        if parent_chunk.metadata.get("parent_id") is not None
    }
    parent_ids = [
        str(document.metadata["parent_id"])
        for document in collapsed_parents
        if document.metadata.get("parent_id") is not None
    ]
    return RetrievalCheckpoint(
        stage_name="collapsed_parents",
        child_ids=[
            str(document.metadata["child_id"])
            for document in collapsed_parents
            if document.metadata.get("child_id") is not None
        ],
        parent_ids=parent_ids,
        contexts=[parent_text_by_id.get(parent_id, "") for parent_id in parent_ids],
        items=[
            {
                "rank": index,
                "parent_id": None
                if document.metadata.get("parent_id") is None
                else str(document.metadata.get("parent_id")),
                "child_id": None
                if document.metadata.get("child_id") is None
                else str(document.metadata.get("child_id")),
                "rerank_score": document.metadata.get("rerank_score"),
                "point_id": document.id,
                "child_text": document.page_content,
                "parent_found": str(document.metadata.get("parent_id")) in parent_text_by_id
                if document.metadata.get("parent_id") is not None
                else False,
                "parent_text": parent_text_by_id.get(str(document.metadata.get("parent_id")))
                if document.metadata.get("parent_id") is not None
                else None,
            }
            for index, document in enumerate(collapsed_parents, start=1)
        ],
    )


def _queries_for_retrieval(
    sample: EvalSample,
    config: ExperimentConfig,
    llm: Any,
    max_queries: int,
) -> list[str]:
    if not config.enable_query_rewrite:
        return [sample.question]

    rewrite_result = rewrite_queries(sample.question, llm, max_queries=max_queries)
    rewritten_queries = list(rewrite_result.rewritten_queries) or [sample.question]
    if sample.question not in rewritten_queries:
        rewritten_queries.insert(0, sample.question)
    return rewritten_queries


def _message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if content is not None:
        return str(content)
    return str(message)


def run_stage_trace(
    *,
    sample: EvalSample,
    config: ExperimentConfig,
    llm: Any,
    client: Any,
    collection_name: str,
    embeddings: Any,
    sparse_embeddings: Any,
    mongo_repository: Any,
    reranker: Any | None = None,
    candidate_limit: int = 30,
    max_queries: int = 4,
) -> StageTrace:
    rewritten_queries = _queries_for_retrieval(sample, config, llm, max_queries)
    active_queries = (
        rewritten_queries if config.enable_multi_query_merge else rewritten_queries[:1]
    )

    retrieval_checkpoints: list[RetrievalCheckpoint] = []
    hit_groups: list[list[HybridRetrievalHit]] = []
    query_hits: list[tuple[str, list[HybridRetrievalHit]]] = []
    for query_text in active_queries:
        hits = query_hybrid_children(
            client=client,
            collection_name=collection_name,
            query_text=query_text,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            top_k=config.top_k_per_query,
            candidate_limit=candidate_limit,
        )
        hit_groups.append(hits)
        query_hits.append((query_text, hits))
        retrieval_checkpoints.append(_hybrid_per_query_checkpoint(query_text, hits))

    merged_hits = merge_hybrid_hits(*hit_groups, candidate_limit=candidate_limit)
    retrieval_checkpoints.append(
        _merged_candidates_checkpoint(query_hits=query_hits, merged_hits=merged_hits)
    )

    merged_documents = [_child_hit_to_document(hit) for hit in merged_hits]
    if config.enable_rerank:
        reranked_candidates = rerank_candidates(
            sample.question,
            merged_documents,
            reranker,
            limit=candidate_limit,
        )
    else:
        reranked_candidates = merged_documents
    retrieval_checkpoints.append(
        _documents_checkpoint("reranked_candidates", reranked_candidates)
    )

    collapsed_parents = collapse_to_parent_hits(
        reranked_candidates,
        limit=config.final_parent_limit,
    )
    parent_ids = [
        str(document.metadata["parent_id"])
        for document in collapsed_parents
        if document.metadata.get("parent_id") is not None
    ]
    parent_chunks = fetch_parent_chunks(parent_ids, mongo_repository)
    retrieval_checkpoints.append(_collapsed_parents_checkpoint(collapsed_parents, parent_chunks))

    prompt = build_answer_prompt(sample.question, parent_chunks)
    final_answer = _message_to_text(llm.invoke(prompt))

    return StageTrace(
        sample=sample,
        rewritten_queries=rewritten_queries,
        retrieval_checkpoints=retrieval_checkpoints,
        final_answer=final_answer,
    )


__all__ = ["run_stage_trace"]
