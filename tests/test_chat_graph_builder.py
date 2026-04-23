from __future__ import annotations

import asyncio
from types import SimpleNamespace

from graphs.chat.models import ChatGraphInput
from graphs.chat.nodes import ChatNodes


def test_chat_graph_runs_linear_pipeline(monkeypatch) -> None:
    execution_order: list[str] = []

    def record_sync(stage: str):
        def runner(self: ChatNodes, state: dict[str, object]) -> dict[str, object]:
            execution_order.append(stage)
            if stage == "response":
                state["response_payload"] = {
                    "answer": state.get("answer", ""),
                    "source_items": state.get("source_items", []),
                }
            return state

        return runner

    def record_async(stage: str):
        async def runner(self: ChatNodes, state: dict[str, object]) -> dict[str, object]:
            execution_order.append(stage)
            if stage == "generate":
                state["answer"] = "graph-answer"
            return state

        return runner

    monkeypatch.setattr(ChatNodes, "prepare_query", record_sync("prepare"))
    monkeypatch.setattr(ChatNodes, "rewrite_query", record_async("rewrite"))
    monkeypatch.setattr(ChatNodes, "retrieve_candidates", record_async("retrieve"))
    monkeypatch.setattr(ChatNodes, "merge_candidates", record_sync("merge"))
    monkeypatch.setattr(ChatNodes, "rerank_candidates", record_async("rerank"))
    monkeypatch.setattr(ChatNodes, "recall_parents", record_async("recall"))
    monkeypatch.setattr(ChatNodes, "generate_answer", record_async("generate"))
    monkeypatch.setattr(ChatNodes, "build_response", record_sync("response"))

    from graphs.chat.builder import build_chat_graph

    app = build_chat_graph(SimpleNamespace())
    result = asyncio.run(app.ainvoke(ChatGraphInput(question="hello", request_id="req-1").model_dump()))

    assert execution_order == [
        "prepare",
        "rewrite",
        "retrieve",
        "merge",
        "rerank",
        "recall",
        "generate",
        "response",
    ]
    assert result["answer"] == "graph-answer"
    assert result["response_payload"] == {"answer": "graph-answer", "source_items": []}
    assert isinstance(result["timings"], dict)
