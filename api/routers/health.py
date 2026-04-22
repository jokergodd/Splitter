from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from api.dependencies import get_runtime
from api.schemas import StatusResponse

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health", response_model=StatusResponse)
async def health() -> StatusResponse:
    return StatusResponse(status="ok")


@router.get("/ready", response_model=StatusResponse)
async def ready(runtime: Any = Depends(get_runtime)) -> StatusResponse:
    mongo_client = runtime.storage_backend.mongo_repository.async_client
    qdrant_client = runtime.storage_backend.qdrant_store.async_client
    await mongo_client.admin.command("ping")
    await qdrant_client.get_collections()
    return StatusResponse(status="ready")
