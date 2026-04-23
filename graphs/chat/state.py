from __future__ import annotations

from typing import Any, Literal, TypedDict

from graphs.chat.models import ChatGraphInput


ChatStageStatus = Literal["pending", "completed", "fallback", "skipped"]


class ChatGraphRequestContext(TypedDict):
    request_id: str | None
    top_k: int
    candidate_limit: int
    max_queries: int
    parent_limit: int


class ChatGraphState(TypedDict):
    question: str
    normalized_question: str
    rewritten_queries: list[str]
    retrieved_children: list[Any]
    per_query_hits: dict[str, list[Any]]
    merged_children: list[Any]
    reranked_children: list[Any]
    parent_chunks: list[Any]
    answer: str
    source_items: list[dict[str, str | None]]
    response_payload: dict[str, Any]
    errors: list[dict[str, str]]
    timings: dict[str, float]
    request_context: ChatGraphRequestContext
    rewrite_status: ChatStageStatus
    rerank_status: ChatStageStatus
    generation_status: ChatStageStatus


def initialize_state(graph_input: ChatGraphInput) -> ChatGraphState:
    return {
        "question": graph_input.question,
        "normalized_question": graph_input.question,
        "rewritten_queries": [],
        "retrieved_children": [],
        "per_query_hits": {},
        "merged_children": [],
        "reranked_children": [],
        "parent_chunks": [],
        "answer": "",
        "source_items": [],
        "response_payload": {},
        "errors": [],
        "timings": {},
        "request_context": {
            "request_id": graph_input.request_id,
            "top_k": graph_input.top_k,
            "candidate_limit": graph_input.candidate_limit,
            "max_queries": graph_input.max_queries,
            "parent_limit": graph_input.parent_limit,
        },
        "rewrite_status": "pending",
        "rerank_status": "pending",
        "generation_status": "pending",
    }
