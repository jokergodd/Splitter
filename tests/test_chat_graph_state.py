from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphs.chat.models import ChatGraphInput
from graphs.chat.state import ChatGraphState, initialize_state


def test_initialize_state_sets_request_context_and_defaults() -> None:
    graph_input = ChatGraphInput(
        question="如何申请出差报销？",
        top_k=6,
        candidate_limit=12,
        max_queries=3,
        parent_limit=4,
        request_id="req-123",
    )

    state = initialize_state(graph_input)

    assert state["question"] == "如何申请出差报销？"
    assert state["normalized_question"] == "如何申请出差报销？"
    assert state["rewritten_queries"] == []
    assert state["retrieved_children"] == []
    assert state["per_query_hits"] == {}
    assert state["merged_children"] == []
    assert state["reranked_children"] == []
    assert state["parent_chunks"] == []
    assert state["answer"] == ""
    assert state["source_items"] == []
    assert state["response_payload"] == {}
    assert state["errors"] == []
    assert state["timings"] == {}
    assert state["request_context"] == {
        "request_id": "req-123",
        "top_k": 6,
        "candidate_limit": 12,
        "max_queries": 3,
        "parent_limit": 4,
    }
    assert state["rewrite_status"] == "pending"
    assert state["rerank_status"] == "pending"
    assert state["generation_status"] == "pending"

    typed_state: ChatGraphState = state
    assert typed_state["normalized_question"] == "如何申请出差报销？"


def test_initialize_state_trims_question_and_keeps_request_id_none() -> None:
    graph_input = ChatGraphInput(question="  请假流程是什么？  ")

    state = initialize_state(graph_input)

    assert state["question"] == "请假流程是什么？"
    assert state["normalized_question"] == "请假流程是什么？"
    assert state["request_context"] == {
        "request_id": None,
        "top_k": 10,
        "candidate_limit": 30,
        "max_queries": 4,
        "parent_limit": 5,
    }


def test_chat_graph_input_rejects_whitespace_only_question() -> None:
    with pytest.raises(ValidationError):
        ChatGraphInput(question="   ")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("top_k", 0),
        ("top_k", -1),
        ("candidate_limit", 0),
        ("candidate_limit", -1),
        ("max_queries", 0),
        ("max_queries", -1),
        ("parent_limit", 0),
        ("parent_limit", -1),
    ],
)
def test_chat_graph_input_rejects_non_positive_numeric_parameters(
    field_name: str, value: int
) -> None:
    payload = {"question": "有效问题", field_name: value}

    with pytest.raises(ValidationError):
        ChatGraphInput(**payload)
