from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from graphs.chat.models import ChatGraphInput
from graphs.chat.nodes import build_nodes
from graphs.chat.state import ChatGraphState, initialize_state


class ChatGraphApp:
    def __init__(self, compiled_graph: Any) -> None:
        self._compiled_graph = compiled_graph

    async def ainvoke(self, payload: ChatGraphInput | dict[str, Any]) -> ChatGraphState:
        graph_input = ChatGraphInput.model_validate(payload)
        initial_state = initialize_state(graph_input)
        return await self._compiled_graph.ainvoke(initial_state)


def build_chat_graph(deps: Any) -> ChatGraphApp:
    nodes = build_nodes(deps)
    graph = StateGraph(ChatGraphState)

    graph.add_node("prepare_query", nodes.prepare_query)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("retrieve_candidates", nodes.retrieve_candidates)
    graph.add_node("merge_candidates", nodes.merge_candidates)
    graph.add_node("rerank_candidates", nodes.rerank_candidates)
    graph.add_node("recall_parents", nodes.recall_parents)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("build_response", nodes.build_response)

    graph.set_entry_point("prepare_query")
    graph.add_edge("prepare_query", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "merge_candidates")
    graph.add_edge("merge_candidates", "rerank_candidates")
    graph.add_edge("rerank_candidates", "recall_parents")
    graph.add_edge("recall_parents", "generate_answer")
    graph.add_edge("generate_answer", "build_response")
    graph.add_edge("build_response", END)

    return ChatGraphApp(graph.compile())


__all__ = ["build_chat_graph", "ChatGraphApp"]
