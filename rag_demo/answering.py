from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from .parent_recall import collapse_to_parent_hits, fetch_parent_chunks, fetch_parent_chunks_async
from .query_rewrite import rewrite_queries, rewrite_queries_async
from .rerank import rerank_candidates
from .retrieval import (
    HybridRetrievalHit,
    merge_hybrid_hits,
    query_hybrid_children_for_queries,
    query_hybrid_children_for_queries_async,
)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    rewritten_queries: list[str] = field(default_factory=list)
    parent_chunks: list[Document] = field(default_factory=list)
    source_items: list[dict[str, str | None]] = field(default_factory=list)


def child_hit_to_document(hit: HybridRetrievalHit) -> Document:
    metadata = dict(hit.payload)
    metadata.setdefault("child_id", hit.child_id)
    metadata.setdefault("retrieval_score", hit.score)
    return Document(
        page_content=str(hit.payload.get("text", "")),
        metadata=metadata,
        id=hit.point_id,
    )


def message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if content is not None:
        return str(content)
    return str(message)


def build_source_items(parent_chunks: Sequence[Document]) -> list[dict[str, str | None]]:
    items: list[dict[str, str | None]] = []
    for parent_chunk in parent_chunks:
        metadata = parent_chunk.metadata
        items.append(
            {
                "parent_id": None
                if metadata.get("parent_id") is None
                else str(metadata.get("parent_id")),
                "source": None if metadata.get("source") is None else str(metadata.get("source")),
                "file_path": None
                if metadata.get("file_path") is None
                else str(metadata.get("file_path")),
            }
        )
    return items


def retrieval_fallback_documents(candidate_documents: Sequence[Document]) -> list[Document]:
    fallback_documents: list[Document] = []
    for candidate in candidate_documents:
        metadata = dict(candidate.metadata)
        retrieval_score = float(metadata.get("retrieval_score", 0.0))
        metadata["rerank_score"] = retrieval_score
        fallback_documents.append(
            Document(
                page_content=candidate.page_content,
                metadata=metadata,
                id=candidate.id,
            )
        )
    fallback_documents.sort(key=lambda doc: float(doc.metadata.get("rerank_score", 0.0)), reverse=True)
    return fallback_documents


def merge_retrieved_child_hit_groups(
    hit_groups: Sequence[Sequence[HybridRetrievalHit]],
    *,
    candidate_limit: int,
) -> list[Document]:
    merged_hits = merge_hybrid_hits(*hit_groups, candidate_limit=candidate_limit)
    return [child_hit_to_document(hit) for hit in merged_hits]


async def retrieve_candidate_documents_async(
    *,
    client: Any,
    collection_name: str,
    query_texts: Sequence[str],
    embeddings: Any,
    sparse_embeddings: Any,
    top_k: int = 10,
    candidate_limit: int = 30,
) -> list[Document]:
    candidate_hits = await query_hybrid_children_for_queries_async(
        client=client,
        collection_name=collection_name,
        query_texts=query_texts,
        embeddings=embeddings,
        sparse_embeddings=sparse_embeddings,
        top_k=top_k,
        candidate_limit=candidate_limit,
    )
    return [child_hit_to_document(hit) for hit in candidate_hits]


def build_answer_prompt(original_query: str, parent_chunks: Sequence[Document]) -> str:
    sections = [
        "You are answering a question using the retrieved parent chunks.",
        f"Original query: {original_query}",
        "",
        "Parent chunks:",
    ]

    for index, parent_chunk in enumerate(parent_chunks, start=1):
        parent_id = parent_chunk.metadata.get("parent_id", index)
        sections.extend(
            [
                f"[Parent {index} | parent_id={parent_id}]",
                parent_chunk.page_content,
                "",
            ]
        )

    sections.extend(
        [
            "Answer the original query only from the parent chunks above.",
            f"Question: {original_query}",
        ]
    )
    return "\n".join(sections).strip()


def answer_query(
    *,
    original_query: str,
    llm: Any,
    client: Any,
    collection_name: str,
    embeddings: Any,
    sparse_embeddings: Any,
    mongo_repository: Any,
    top_k: int = 10,
    candidate_limit: int = 30,
    max_queries: int = 4,
    reranker: Any | None = None,
    parent_limit: int = 5,
) -> AnswerResult:
    rewrite_result = rewrite_queries(original_query, llm, max_queries=max_queries)
    candidate_hits = query_hybrid_children_for_queries(
        client=client,
        collection_name=collection_name,
        query_texts=rewrite_result.rewritten_queries,
        embeddings=embeddings,
        sparse_embeddings=sparse_embeddings,
        top_k=top_k,
        candidate_limit=candidate_limit,
    )

    candidate_documents = [child_hit_to_document(hit) for hit in candidate_hits]
    reranked_documents = rerank_candidates(
        original_query,
        candidate_documents,
        reranker,
        limit=candidate_limit,
    )
    parent_hit_documents = collapse_to_parent_hits(reranked_documents, limit=parent_limit)
    parent_ids = [
        str(parent_hit.metadata["parent_id"])
        for parent_hit in parent_hit_documents
        if parent_hit.metadata.get("parent_id") is not None
    ]
    parent_chunks = fetch_parent_chunks(parent_ids, mongo_repository)
    prompt = build_answer_prompt(original_query, parent_chunks)
    llm_result = llm.invoke(prompt)
    return AnswerResult(
        answer=message_to_text(llm_result),
        rewritten_queries=list(rewrite_result.rewritten_queries),
        parent_chunks=parent_chunks,
        source_items=build_source_items(parent_chunks),
    )


async def answer_query_async(
    *,
    original_query: str,
    llm: Any,
    client: Any,
    collection_name: str,
    embeddings: Any,
    sparse_embeddings: Any,
    mongo_repository: Any,
    top_k: int = 10,
    candidate_limit: int = 30,
    max_queries: int = 4,
    reranker: Any | None = None,
    parent_limit: int = 5,
) -> AnswerResult:
    rewrite_result = await rewrite_queries_async(original_query, llm, max_queries=max_queries)
    candidate_hits = await query_hybrid_children_for_queries_async(
        client=client,
        collection_name=collection_name,
        query_texts=rewrite_result.rewritten_queries,
        embeddings=embeddings,
        sparse_embeddings=sparse_embeddings,
        top_k=top_k,
        candidate_limit=candidate_limit,
    )

    candidate_documents = [child_hit_to_document(hit) for hit in candidate_hits]
    reranked_documents = await asyncio.to_thread(
        rerank_candidates,
        original_query,
        candidate_documents,
        reranker,
        candidate_limit,
    )
    parent_hit_documents = collapse_to_parent_hits(reranked_documents, limit=parent_limit)
    parent_ids = [
        str(parent_hit.metadata["parent_id"])
        for parent_hit in parent_hit_documents
        if parent_hit.metadata.get("parent_id") is not None
    ]
    parent_chunks = await fetch_parent_chunks_async(parent_ids, mongo_repository)
    prompt = build_answer_prompt(original_query, parent_chunks)
    if hasattr(llm, "ainvoke"):
        llm_result = await llm.ainvoke(prompt)
    else:
        llm_result = await asyncio.to_thread(llm.invoke, prompt)
    return AnswerResult(
        answer=message_to_text(llm_result),
        rewritten_queries=list(rewrite_result.rewritten_queries),
        parent_chunks=parent_chunks,
        source_items=build_source_items(parent_chunks),
    )


__all__ = [
    "AnswerResult",
    "answer_query",
    "answer_query_async",
    "build_answer_prompt",
    "child_hit_to_document",
    "build_source_items",
    "message_to_text",
    "merge_retrieved_child_hit_groups",
    "retrieve_candidate_documents_async",
    "retrieval_fallback_documents",
]
