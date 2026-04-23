from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from rag_demo import answering
from rag_demo.parent_recall import collapse_to_parent_hits, fetch_parent_chunks_async
from rag_demo.query_rewrite import rewrite_queries_async
from rag_demo.rerank import rerank_candidates


def _mark_timing(state: dict[str, Any], stage: str, started_at: float) -> None:
    state["timings"][stage] = round((time.perf_counter() - started_at) * 1000, 3)


def _append_error(state: dict[str, Any], *, stage: str, error: Exception) -> None:
    message = str(error) or repr(error)
    state["errors"].append({"stage": stage, "error": message})


def _get_request_limit(state: dict[str, Any], key: str, default: int) -> int:
    value = state["request_context"].get(key, default)
    return int(value)


@dataclass(slots=True)
class ChatNodes:
    deps: Any

    def prepare_query(self, state: dict[str, Any]) -> dict[str, Any]:
        state["normalized_question"] = state["question"].strip()
        return state

    async def rewrite_query(self, state: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        question = state["normalized_question"]
        max_queries = _get_request_limit(state, "max_queries", 4)
        rewrite_impl = getattr(self.deps, "rewrite_queries", None)
        if rewrite_impl is None:
            rewrite_impl = lambda *, question, max_queries: rewrite_queries_async(  # noqa: E731
                question,
                self.deps.llm,
                max_queries=max_queries,
            )

        try:
            result = await rewrite_impl(question=question, max_queries=max_queries)
            rewritten_queries = list(getattr(result, "rewritten_queries", []) or [])
            state["rewritten_queries"] = rewritten_queries or [question]
            state["rewrite_status"] = "completed"
        except Exception as exc:
            state["rewritten_queries"] = [question]
            state["rewrite_status"] = "fallback"
            _append_error(state, stage="rewrite_query", error=exc)

        _mark_timing(state, "rewrite_query", started_at)
        return state

    async def retrieve_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        query_texts = state["rewritten_queries"] or [state["normalized_question"]]
        top_k = _get_request_limit(state, "top_k", 10)
        candidate_limit = _get_request_limit(state, "candidate_limit", 30)

        state["retrieved_children"] = await answering.retrieve_candidate_documents_async(
            client=self.deps.qdrant_client,
            collection_name=self.deps.collection_name,
            query_texts=query_texts,
            embeddings=self.deps.dense_embeddings,
            sparse_embeddings=self.deps.sparse_embeddings,
            top_k=top_k,
            candidate_limit=candidate_limit,
        )
        state["per_query_hits"] = {}
        _mark_timing(state, "retrieve_candidates", started_at)
        return state

    def merge_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        state["merged_children"] = list(state["retrieved_children"])
        _mark_timing(state, "merge_candidates", started_at)
        return state

    async def rerank_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        candidates = state["merged_children"]
        candidate_limit = _get_request_limit(state, "candidate_limit", 30)

        try:
            reranked = await asyncio.to_thread(
                rerank_candidates,
                state["normalized_question"],
                candidates,
                getattr(self.deps, "reranker", None),
                candidate_limit,
            )
            state["reranked_children"] = reranked
            state["rerank_status"] = "completed"
        except Exception as exc:
            state["reranked_children"] = answering.retrieval_fallback_documents(candidates)
            state["rerank_status"] = "fallback"
            _append_error(state, stage="rerank_candidates", error=exc)

        _mark_timing(state, "rerank_candidates", started_at)
        return state

    async def recall_parents(self, state: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        reranked_children = state["reranked_children"] or state["merged_children"]
        parent_limit = _get_request_limit(state, "parent_limit", 5)
        parent_hits = collapse_to_parent_hits(reranked_children, limit=parent_limit)
        parent_ids = [
            str(parent_hit.metadata["parent_id"])
            for parent_hit in parent_hits
            if parent_hit.metadata.get("parent_id") is not None
        ]
        state["parent_chunks"] = await fetch_parent_chunks_async(parent_ids, self.deps.mongo_repository)
        state["source_items"] = answering.build_source_items(state["parent_chunks"])
        _mark_timing(state, "recall_parents", started_at)
        return state

    async def generate_answer(self, state: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        prompt = answering.build_answer_prompt(state["normalized_question"], state["parent_chunks"])

        try:
            if hasattr(self.deps.llm, "ainvoke"):
                llm_result = await self.deps.llm.ainvoke(prompt)
            else:
                llm_result = await asyncio.to_thread(self.deps.llm.invoke, prompt)
            state["answer"] = answering.message_to_text(llm_result)
            state["generation_status"] = "completed"
            return state
        finally:
            _mark_timing(state, "generate_answer", started_at)

    def build_response(self, state: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        source_items = state["source_items"] or answering.build_source_items(state["parent_chunks"])
        state["source_items"] = source_items
        state["response_payload"] = {
            "answer": state["answer"],
            "source_items": source_items,
        }
        _mark_timing(state, "build_response", started_at)
        return state


def build_nodes(deps: Any) -> ChatNodes:
    return ChatNodes(deps=deps)


__all__ = ["ChatNodes", "build_nodes"]
