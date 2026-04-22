from __future__ import annotations
import logging
import time
from pathlib import Path

from pymongo import errors as pymongo_errors
from qdrant_client.common.client_exceptions import QdrantException

from rag_demo.pipeline import (
    BatchResult,
    PipelineConfig,
    PipelineResult,
    run_batch_pipeline_async,
    run_document_pipeline_async,
)

from runtime.container import Runtime
from services.exceptions import CollectionNotReadyError, DependencyUnavailableError
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


def _normalize_ingest_error(exc: Exception, collection_name: str) -> Exception:
    if isinstance(exc, (CollectionNotReadyError, DependencyUnavailableError)):
        return exc
    if isinstance(exc, (QdrantException, ApiException, ResponseHandlingException, UnexpectedResponse)):
        if collection_name and _is_collection_not_ready(exc, collection_name):
            return CollectionNotReadyError(collection_name)
        return DependencyUnavailableError("Qdrant is unavailable", dependency="qdrant")
    if isinstance(exc, pymongo_errors.PyMongoError):
        return DependencyUnavailableError("MongoDB is unavailable", dependency="mongodb")
    return exc


class IngestService:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    async def ingest_file(
        self,
        *,
        file_path: str | Path,
        config: PipelineConfig | None = None,
    ) -> PipelineResult:
        return await ingest_file(file_path, self.runtime, config=config)

    async def ingest_batch(
        self,
        *,
        data_dir: str | Path,
        pipeline_config: PipelineConfig | None = None,
    ) -> BatchResult:
        return await ingest_batch(
            data_dir,
            self.runtime,
            pipeline_config=pipeline_config,
        )


async def ingest_file(
    file_path: str | Path,
    runtime: Runtime,
    *,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    pipeline_config = config or PipelineConfig()
    collection_name = _collection_name(runtime) or ""
    started_at = time.perf_counter()
    logger.info(
        "ingest.file.started",
        extra=structured_extra(
            "ingest.file.started",
            file_path=file_path,
        ),
    )
    try:
        result = await run_document_pipeline_async(
            file_path=file_path,
            config=pipeline_config,
            embeddings=runtime.dense_embeddings,
            storage_backend=runtime.storage_backend,
        )
    except Exception as exc:
        normalized_exc = _normalize_ingest_error(exc, collection_name)
        logger.error(
            "ingest.file.failed",
            extra=structured_extra(
                "ingest.file.failed",
                file_path=file_path,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                error=str(normalized_exc) if str(normalized_exc) else repr(normalized_exc),
            ),
        )
        raise normalized_exc from exc
    logger.info(
        "ingest.file.completed",
        extra=structured_extra(
            "ingest.file.completed",
            file_path=file_path,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
        ),
    )
    return result


async def ingest_batch(
    directory: str | Path,
    runtime: Runtime,
    *,
    pipeline_config: PipelineConfig | None = None,
) -> BatchResult:
    collection_name = _collection_name(runtime) or ""
    started_at = time.perf_counter()
    logger.info(
        "ingest.batch.started",
        extra=structured_extra(
            "ingest.batch.started",
            data_dir=directory,
        ),
    )
    try:
        result = await run_batch_pipeline_async(
            directory=directory,
            embeddings=runtime.dense_embeddings,
            pipeline_config=pipeline_config or PipelineConfig(),
            storage_backend=runtime.storage_backend,
        )
    except Exception as exc:
        normalized_exc = _normalize_ingest_error(exc, collection_name)
        logger.error(
            "ingest.batch.failed",
            extra=structured_extra(
                "ingest.batch.failed",
                data_dir=directory,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                error=str(normalized_exc) if str(normalized_exc) else repr(normalized_exc),
            ),
        )
        raise normalized_exc from exc
    logger.info(
        "ingest.batch.completed",
        extra=structured_extra(
            "ingest.batch.completed",
            data_dir=directory,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
        ),
    )
    return result


__all__ = ["IngestService", "ingest_batch", "ingest_file"]
