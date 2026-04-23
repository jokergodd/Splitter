from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_chat_service
from api.schemas import ChatQueryRequest, ChatQueryResponse

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/query", response_model=ChatQueryResponse, response_model_exclude_none=True)
async def query_chat(
    payload: ChatQueryRequest,
    request: Request,
    chat_service: Any = Depends(get_chat_service),
) -> ChatQueryResponse:
    answer_kwargs: dict[str, Any] = {"question": payload.question}
    answer_signature = inspect.signature(chat_service.answer)
    if "request_id" in answer_signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in answer_signature.parameters.values()
    ):
        answer_kwargs["request_id"] = getattr(request.state, "request_id", None)
    result = await chat_service.answer(**answer_kwargs)
    return ChatQueryResponse.from_result(result)
