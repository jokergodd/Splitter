from __future__ import annotations

import tempfile
from pathlib import Path

from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import get_ingest_service
from api.schemas import IngestBatchRequest, IngestResponse

router = APIRouter(prefix="/v1/ingest", tags=["ingest"])


@router.post("/file", response_model=IngestResponse, response_model_exclude_none=True)
async def ingest_file(
    file: UploadFile = File(...),
    ingest_service: Any = Depends(get_ingest_service),
) -> IngestResponse:
    suffix = Path(file.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)
        handle.write(await file.read())
    try:
        result = await ingest_service.ingest_file(file_path=str(temp_path))
        return IngestResponse.from_result(result, mode="file")
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/batch", response_model=IngestResponse, response_model_exclude_none=True)
async def ingest_batch(
    payload: IngestBatchRequest,
    ingest_service: Any = Depends(get_ingest_service),
) -> IngestResponse:
    result = await ingest_service.ingest_batch(data_dir=payload.data_dir)
    return IngestResponse.from_result(result, mode="batch")
