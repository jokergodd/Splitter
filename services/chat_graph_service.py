from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from graphs.chat.builder import build_chat_graph
from graphs.chat.models import ChatGraphInput
from rag_demo.answering import AnswerResult

_GRAPH_APP_ATTR = "_chat_graph_app"


def _coerce_answer_result(result: Any) -> AnswerResult:
    if isinstance(result, AnswerResult):
        return result
    if isinstance(result, Mapping):
        payload = result.get("response_payload")
        if isinstance(payload, Mapping):
            answer = payload.get("answer", "")
            source_items = list(payload.get("source_items") or [])
        else:
            answer = result.get("answer", "")
            source_items = list(result.get("source_items") or [])
        return AnswerResult(
            answer="" if answer is None else str(answer),
            rewritten_queries=list(result.get("rewritten_queries") or []),
            parent_chunks=list(result.get("parent_chunks") or []),
            source_items=source_items,
        )
    return AnswerResult(answer="")


def _build_graph_dependencies(runtime: Any) -> SimpleNamespace:
    qdrant_store = runtime.storage_backend.qdrant_store
    return SimpleNamespace(
        llm=runtime.llm,
        qdrant_client=qdrant_store.async_client,
        collection_name=qdrant_store.collection_name,
        dense_embeddings=runtime.dense_embeddings,
        sparse_embeddings=runtime.sparse_embeddings,
        mongo_repository=runtime.storage_backend.mongo_repository,
        reranker=runtime.reranker,
    )


def _get_or_create_graph_app(runtime: Any) -> Any:
    graph_app = getattr(runtime, _GRAPH_APP_ATTR, None)
    if graph_app is None:
        graph_app = build_chat_graph(_build_graph_dependencies(runtime))
        setattr(runtime, _GRAPH_APP_ATTR, graph_app)
    return graph_app


class ChatGraphService:
    def __init__(
        self,
        runtime: Any | None = None,
        graph_app: Any | None = None,
        runtime_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.graph_app = graph_app
        self._runtime_factory = runtime_factory

    def _ensure_runtime(self) -> Any:
        if self.runtime is None:
            if self._runtime_factory is None:
                raise RuntimeError("ChatGraphService requires runtime or runtime_factory")
            self.runtime = self._runtime_factory()
        return self.runtime

    def _ensure_graph_app(self) -> Any:
        if self.graph_app is None:
            self.graph_app = _get_or_create_graph_app(self._ensure_runtime())
        return self.graph_app

    async def answer(
        self,
        *,
        question: str,
        top_k: int = 10,
        candidate_limit: int = 30,
        max_queries: int = 4,
        parent_limit: int = 5,
        request_id: str | None = None,
    ) -> AnswerResult:
        payload = ChatGraphInput(
            question=question,
            top_k=top_k,
            candidate_limit=candidate_limit,
            max_queries=max_queries,
            parent_limit=parent_limit,
            request_id=request_id,
        ).model_dump()
        result = await self._ensure_graph_app().ainvoke(payload)
        return _coerce_answer_result(result)


__all__ = ["ChatGraphService"]
