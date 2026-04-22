from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from services.exceptions import TaskNotFoundError
from services.task_models import TaskRecord, TaskType
from services.task_registry import TaskRegistry
from services.logging_utils import normalize_log_value, structured_extra


def _error_message(exc: Exception) -> str:
    message = str(exc)
    return message if message else repr(exc)


logger = logging.getLogger(__name__)


class TaskService:
    def __init__(
        self,
        ingest_service: Any | None = None,
        runtime: Any | None = None,
        *,
        registry: TaskRegistry | None = None,
        executor: ThreadPoolExecutor | None = None,
        max_workers: int = 4,
    ) -> None:
        if ingest_service is None:
            if runtime is None:
                raise ValueError("TaskService requires ingest_service or runtime")
            from services.ingest_service import IngestService

            ingest_service = IngestService(runtime)

        self.ingest_service = ingest_service
        self.registry = registry or TaskRegistry()
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="task-service")
        self._owns_executor = executor is None

    def submit_file_task(self, *, file_path: str | Path, config: Any | None = None) -> str:
        task = self.registry.register(TaskType.INGEST_FILE)
        logger.info(
            "task.submitted",
            extra=structured_extra(
                "task.submitted",
                task_id=task.task_id,
                task_type=task.task_type,
                file_path=file_path,
            ),
        )
        self._executor.submit(self._run_file_task, task.task_id, file_path, config)
        return task.task_id

    def submit_batch_task(self, *, data_dir: str | Path, pipeline_config: Any | None = None) -> str:
        task = self.registry.register(TaskType.INGEST_BATCH)
        logger.info(
            "task.submitted",
            extra=structured_extra(
                "task.submitted",
                task_id=task.task_id,
                task_type=task.task_type,
                data_dir=data_dir,
            ),
        )
        self._executor.submit(self._run_batch_task, task.task_id, data_dir, pipeline_config)
        return task.task_id

    def get_task_status(self, task_id: str) -> TaskRecord | None:
        return self.registry.get(task_id)

    async def submit_ingest_file(self, *, file_path: str | Path, config: Any | None = None) -> dict[str, str]:
        return {"task_id": self.submit_file_task(file_path=file_path, config=config)}

    async def submit_ingest_batch(
        self,
        *,
        data_dir: str | Path,
        pipeline_config: Any | None = None,
    ) -> dict[str, str]:
        return {"task_id": self.submit_batch_task(data_dir=data_dir, pipeline_config=pipeline_config)}

    async def get_task(self, *, task_id: str) -> TaskRecord | None:
        task = self.get_task_status(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return task

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=False)

    def _run_file_task(self, task_id: str, file_path: str | Path, config: Any | None) -> None:
        started_at = time.perf_counter()
        self.registry.mark_running(task_id)
        logger.info(
            "task.started",
            extra=structured_extra(
                "task.started",
                task_id=task_id,
                task_type=TaskType.INGEST_FILE,
                file_path=file_path,
            ),
        )
        try:
            result = asyncio.run(self.ingest_service.ingest_file(file_path=file_path, config=config))
        except Exception as exc:  # pragma: no cover - exercised via tests
            self.registry.mark_failed(task_id, _error_message(exc))
            logger.error(
                "task.failed",
                extra=structured_extra(
                    "task.failed",
                    task_id=task_id,
                    task_type=TaskType.INGEST_FILE,
                    file_path=file_path,
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                    error=_error_message(exc),
                ),
            )
            return
        self.registry.mark_succeeded(task_id, normalize_log_value(result))
        logger.info(
            "task.succeeded",
            extra=structured_extra(
                "task.succeeded",
                task_id=task_id,
                task_type=TaskType.INGEST_FILE,
                file_path=file_path,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            ),
        )

    def _run_batch_task(self, task_id: str, data_dir: str | Path, pipeline_config: Any | None) -> None:
        started_at = time.perf_counter()
        self.registry.mark_running(task_id)
        logger.info(
            "task.started",
            extra=structured_extra(
                "task.started",
                task_id=task_id,
                task_type=TaskType.INGEST_BATCH,
                data_dir=data_dir,
            ),
        )
        try:
            result = asyncio.run(self.ingest_service.ingest_batch(data_dir=data_dir, pipeline_config=pipeline_config))
        except Exception as exc:  # pragma: no cover - exercised via tests
            self.registry.mark_failed(task_id, _error_message(exc))
            logger.error(
                "task.failed",
                extra=structured_extra(
                    "task.failed",
                    task_id=task_id,
                    task_type=TaskType.INGEST_BATCH,
                    data_dir=data_dir,
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                    error=_error_message(exc),
                ),
            )
            return
        self.registry.mark_succeeded(task_id, normalize_log_value(result))
        logger.info(
            "task.succeeded",
            extra=structured_extra(
                "task.succeeded",
                task_id=task_id,
                task_type=TaskType.INGEST_BATCH,
                data_dir=data_dir,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            ),
        )

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self.close()


__all__ = ["TaskService"]
