from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from api.dependencies import get_chat_service
from api.schemas import ChatQueryRequest, ChatQueryResponse

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/query", response_model=ChatQueryResponse, response_model_exclude_none=True)
async def query_chat(
    payload: ChatQueryRequest,
    chat_service: Any = Depends(get_chat_service),
) -> ChatQueryResponse:
    result = await chat_service.answer(question=payload.question)
    return ChatQueryResponse.from_result(result)
