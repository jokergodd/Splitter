from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import get_task_service
from api.schemas import IngestBatchRequest, TaskSubmissionResponse, TaskStatusResponse
from services.exceptions import TaskNotFoundError

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


@router.post("/ingest/file", response_model=TaskSubmissionResponse)
async def submit_ingest_file(
    file: UploadFile = File(...),
    task_service: Any = Depends(get_task_service),
) -> TaskSubmissionResponse:
    suffix = Path(file.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)
        handle.write(await file.read())

    # The task service owns the staged file once the task is submitted.
    result = await task_service.submit_ingest_file(file_path=str(temp_path))
    return TaskSubmissionResponse.from_result(result)


@router.post("/ingest/batch", response_model=TaskSubmissionResponse)
async def submit_ingest_batch(
    payload: IngestBatchRequest,
    task_service: Any = Depends(get_task_service),
) -> TaskSubmissionResponse:
    result = await task_service.submit_ingest_batch(data_dir=payload.data_dir)
    return TaskSubmissionResponse.from_result(result)


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: str,
    task_service: Any = Depends(get_task_service),
) -> TaskStatusResponse:
    result = await task_service.get_task(task_id=task_id)
    return TaskStatusResponse.from_result(result, task_id=task_id)
