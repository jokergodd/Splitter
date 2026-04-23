from __future__ import annotations

import logging
import time
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from pymongo import errors as pymongo_errors
from qdrant_client.common.client_exceptions import QdrantException

from runtime.container import Runtime
from services.chat_graph_service import ChatGraphService
from services.exceptions import CollectionNotReadyError, DependencyUnavailableError, NoContextRetrievedError
from services.logging_utils import structured_extra

try:
    from qdrant_client.http.exceptions import ApiException, ResponseHandlingException, UnexpectedResponse
except ImportError:  # pragma: no cover - defensive fallback for alternate qdrant versions
    ApiException = ResponseHandlingException = UnexpectedResponse = QdrantException


logger = logging.getLogger(__name__)


def _collection_name(runtime: Runtime) -> str | None:
    storage_backend = getattr(runtime, "storage_backend", None)
    return getattr(getattr(storage_backend, "qdrant_store", None), "collection_name", None)


def _is_collection_not_ready(exc: Exception, collection_name: str) -> bool:
    message = str(exc).lower()
    normalized_collection_name = collection_name.lower()
    return (
        normalized_collection_name in message
        and "collection" in message
        and ("doesn't exist" in message or "not found" in message or "does not exist" in message)
    )


def _normalize_chat_error(exc: Exception, collection_name: str) -> Exception:
    if isinstance(exc, (CollectionNotReadyError, DependencyUnavailableError, NoContextRetrievedError)):
        return exc
    if isinstance(exc, (QdrantException, ApiException, ResponseHandlingException, UnexpectedResponse)):
        if collection_name and _is_collection_not_ready(exc, collection_name):
            return CollectionNotReadyError(collection_name)
        return DependencyUnavailableError("Qdrant is unavailable", dependency="qdrant")
    if isinstance(exc, pymongo_errors.PyMongoError):
        return DependencyUnavailableError("MongoDB is unavailable", dependency="mongodb")
    return exc


def _has_context(result: Any) -> bool:
    if isinstance(result, Mapping):
        return bool(result.get("parent_chunks") or result.get("source_items"))
    if hasattr(result, "parent_chunks") or hasattr(result, "source_items"):
        return bool(getattr(result, "parent_chunks", None) or getattr(result, "source_items", None))
    return True


async def answer_query_async(
    *,
    question: str,
    runtime: Runtime,
    top_k: int = 10,
    candidate_limit: int = 30,
    max_queries: int = 4,
    parent_limit: int = 5,
    request_id: str | None = None,
) -> Any:
    answer_kwargs = {
        "question": question,
        "top_k": top_k,
        "candidate_limit": candidate_limit,
        "max_queries": max_queries,
        "parent_limit": parent_limit,
    }
    if request_id is not None:
        answer_kwargs["request_id"] = request_id
    return await ChatGraphService(runtime).answer(
        **answer_kwargs,
    )


async def _answer_with_compat(
    *,
    question: str,
    runtime: Runtime,
    top_k: int,
    candidate_limit: int,
    max_queries: int,
    parent_limit: int,
    execute: Any,
) -> Any:
    started_at = time.perf_counter()
    logger.info(
        "chat.answer.started",
        extra=structured_extra(
            "chat.answer.started",
            question=question,
            top_k=top_k,
            candidate_limit=candidate_limit,
            max_queries=max_queries,
            parent_limit=parent_limit,
        ),
    )
    collection_name = _collection_name(runtime)
    try:
        result = await execute()
    except Exception as exc:
        normalized_exc = _normalize_chat_error(exc, collection_name)
        logger.error(
            "chat.answer.failed",
            extra=structured_extra(
                "chat.answer.failed",
                question=question,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                error=str(normalized_exc) if str(normalized_exc) else repr(normalized_exc),
            ),
        )
        raise normalized_exc from exc

    if not _has_context(result):
        normalized_exc = NoContextRetrievedError(question)
        logger.error(
            "chat.answer.failed",
            extra=structured_extra(
                "chat.answer.failed",
                question=question,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                error=str(normalized_exc),
            ),
        )
        raise normalized_exc

    logger.info(
        "chat.answer.completed",
        extra=structured_extra(
            "chat.answer.completed",
            question=question,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
        ),
    )
    return result


class ChatService:
    def __init__(
        self,
        runtime: Runtime | None = None,
        runtime_factory: Callable[[], Runtime] | None = None,
    ):
        self.runtime = runtime
        self._runtime_factory = runtime_factory
        self._delegate = ChatGraphService(runtime) if runtime is not None else None

    def _ensure_runtime(self) -> Runtime:
        if self.runtime is None:
            if self._runtime_factory is None:
                raise RuntimeError("ChatService requires runtime or runtime_factory")
            self.runtime = self._runtime_factory()
        return self.runtime

    def _ensure_delegate(self) -> ChatGraphService:
        if self._delegate is None:
            self._delegate = ChatGraphService(self._ensure_runtime())
        return self._delegate

    async def answer(
        self,
        *,
        question: str,
        top_k: int = 10,
        candidate_limit: int = 30,
        max_queries: int = 4,
        parent_limit: int = 5,
        request_id: str | None = None,
    ) -> Any:
        runtime = self._ensure_runtime()
        execute_kwargs = {
            "question": question,
            "top_k": top_k,
            "candidate_limit": candidate_limit,
            "max_queries": max_queries,
            "parent_limit": parent_limit,
        }
        if request_id is not None:
            execute_kwargs["request_id"] = request_id
        return await _answer_with_compat(
            question=question,
            runtime=runtime,
            top_k=top_k,
            candidate_limit=candidate_limit,
            max_queries=max_queries,
            parent_limit=parent_limit,
            execute=lambda: self._ensure_delegate().answer(**execute_kwargs),
        )


async def answer(
    question: str,
    runtime: Runtime,
    *,
    top_k: int = 10,
    candidate_limit: int = 30,
    max_queries: int = 4,
    parent_limit: int = 5,
    request_id: str | None = None,
) -> Any:
    execute_kwargs = {
        "question": question,
        "runtime": runtime,
        "top_k": top_k,
        "candidate_limit": candidate_limit,
        "max_queries": max_queries,
        "parent_limit": parent_limit,
    }
    if request_id is not None:
        execute_kwargs["request_id"] = request_id
    return await _answer_with_compat(
        question=question,
        runtime=runtime,
        top_k=top_k,
        candidate_limit=candidate_limit,
        max_queries=max_queries,
        parent_limit=parent_limit,
        execute=lambda: answer_query_async(**execute_kwargs),
    )


__all__ = ["ChatService", "answer"]
