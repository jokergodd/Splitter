from __future__ import annotations

import logging
import time

from pymongo import errors as pymongo_errors
from qdrant_client.common.client_exceptions import QdrantException

from rag_demo.answering import AnswerResult, answer_query_async

from runtime.container import Runtime
from services.exceptions import CollectionNotReadyError, DependencyUnavailableError, NoContextRetrievedError
from services.logging_utils import structured_extra

try:
    from qdrant_client.http.exceptions import ApiException, ResponseHandlingException, UnexpectedResponse
except ImportError:  # pragma: no cover - defensive fallback for alternate qdrant versions
    ApiException = ResponseHandlingException = UnexpectedResponse = QdrantException


logger = logging.getLogger(__name__)


def _collection_name(runtime: Runtime) -> str | None:
    return getattr(getattr(runtime.storage_backend, "qdrant_store", None), "collection_name", None)


def _is_collection_not_ready(exc: Exception, collection_name: str) -> bool:
    message = str(exc)
    return collection_name in message and "Collection" in message and ("doesn't exist" in message or "Not found" in message)


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


class ChatService:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    async def answer(
        self,
        *,
        question: str,
        top_k: int = 10,
        candidate_limit: int = 30,
        max_queries: int = 4,
        parent_limit: int = 5,
    ) -> AnswerResult:
        return await answer(
            question,
            self.runtime,
            top_k=top_k,
            candidate_limit=candidate_limit,
            max_queries=max_queries,
            parent_limit=parent_limit,
        )


async def answer(
    question: str,
    runtime: Runtime,
    *,
    top_k: int = 10,
    candidate_limit: int = 30,
    max_queries: int = 4,
    parent_limit: int = 5,
) -> AnswerResult:
    hybrid_store = runtime.storage_backend.qdrant_store
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
        result = await answer_query_async(
            original_query=question,
            llm=runtime.llm,
            client=hybrid_store.async_client,
            collection_name=hybrid_store.collection_name,
            embeddings=runtime.dense_embeddings,
            sparse_embeddings=runtime.sparse_embeddings,
            mongo_repository=runtime.storage_backend.mongo_repository,
            top_k=top_k,
            candidate_limit=candidate_limit,
            max_queries=max_queries,
            reranker=runtime.reranker,
            parent_limit=parent_limit,
        )
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

    if not result.parent_chunks:
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


__all__ = ["ChatService", "answer"]
