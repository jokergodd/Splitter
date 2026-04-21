from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from .parent_recall import collapse_to_parent_hits, fetch_parent_chunks
from .query_rewrite import rewrite_queries
from .rerank import rerank_candidates
from .retrieval import HybridRetrievalHit, query_hybrid_children_for_queries


@dataclass(slots=True)
class AnswerResult:
    answer: str
    rewritten_queries: list[str] = field(default_factory=list)
    parent_chunks: list[Document] = field(default_factory=list)
    source_items: list[dict[str, str | None]] = field(default_factory=list)


def _child_hit_to_document(hit: HybridRetrievalHit) -> Document:
    metadata = dict(hit.payload)
    metadata.setdefault("child_id", hit.child_id)
    metadata.setdefault("retrieval_score", hit.score)
    return Document(
        page_content=str(hit.payload.get("text", "")),
        metadata=metadata,
        id=hit.point_id,
    )


def _message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if content is not None:
        return str(content)
    return str(message)


def _source_items(parent_chunks: Sequence[Document]) -> list[dict[str, str | None]]:
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

    candidate_documents = [_child_hit_to_document(hit) for hit in candidate_hits]
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
        answer=_message_to_text(llm_result),
        rewritten_queries=list(rewrite_result.rewritten_queries),
        parent_chunks=parent_chunks,
        source_items=_source_items(parent_chunks),
    )


__all__ = ["AnswerResult", "answer_query", "build_answer_prompt"]
