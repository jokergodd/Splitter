from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from typing import Optional
from uuid import uuid4

REQUEST_ID_HEADER = "X-Request-ID"
_request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id() or "-"
        return True


def generate_request_id() -> str:
    return str(uuid4())


def get_request_id() -> str | None:
    return _request_id_context.get()


def set_request_id(request_id: str | None) -> Token[Optional[str]]:
    return _request_id_context.set(request_id)


def reset_request_id(token: Token[Optional[str]]) -> None:
    _request_id_context.reset(token)


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name or "api")
    if not any(isinstance(filter_, RequestContextFilter) for filter_ in logger.filters):
        logger.addFilter(RequestContextFilter())
    return logger
