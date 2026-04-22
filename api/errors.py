from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pymongo import errors as pymongo_errors
from qdrant_client.common.client_exceptions import QdrantException

from api.logging import REQUEST_ID_HEADER, get_request_id
from services.exceptions import DomainError

try:
    from qdrant_client.http.exceptions import ApiException, ResponseHandlingException, UnexpectedResponse
except ImportError:  # pragma: no cover - defensive fallback for alternate qdrant versions
    ApiException = ResponseHandlingException = UnexpectedResponse = QdrantException


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any = Field(default_factory=dict)

def build_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    request_id: str | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        code=code,
        message=message,
        details={} if details is None else details,
    )
    response = JSONResponse(status_code=status_code, content=jsonable_encoder(payload))
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response


def _message_for_value_error(exc: ValueError) -> str:
    return str(exc) or "Bad request"


def _message_for_file_not_found(exc: FileNotFoundError) -> str:
    return exc.strerror or "File not found"


def _message_for_not_a_directory(exc: NotADirectoryError) -> str:
    return exc.strerror or "Directory not found"


def _message_for_dependency_unavailable(_: Exception) -> str:
    return "Required dependency is unavailable"


def _map_exception(exc: Exception) -> tuple[int, str, str, Any]:
    if isinstance(exc, RequestValidationError):
        return 422, "VALIDATION_ERROR", "Request validation failed", exc.errors()
    if isinstance(exc, DomainError):
        return exc.status_code, exc.code, exc.message, exc.details
    if isinstance(exc, FileNotFoundError):
        return 404, "FILE_NOT_FOUND", _message_for_file_not_found(exc), {}
    if isinstance(exc, NotADirectoryError):
        return 404, "DIRECTORY_NOT_FOUND", _message_for_not_a_directory(exc), {}
    if isinstance(exc, ValueError):
        return 400, "BAD_REQUEST", _message_for_value_error(exc), {}
    if isinstance(exc, (pymongo_errors.PyMongoError, QdrantException, ApiException, ResponseHandlingException, UnexpectedResponse)):
        return 503, "DEPENDENCY_UNAVAILABLE", _message_for_dependency_unavailable(exc), {}
    return 500, "INTERNAL_ERROR", "Unexpected server error", {}


async def _handle_exception(request: Request, exc: Exception) -> JSONResponse:
    status_code, code, message, details = _map_exception(exc)
    request_id = getattr(request.state, "request_id", None) or get_request_id()
    return build_error_response(
        status_code=status_code,
        code=code,
        message=message,
        details=details,
        request_id=request_id,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _handle_exception)
    app.add_exception_handler(DomainError, _handle_exception)
    app.add_exception_handler(FileNotFoundError, _handle_exception)
    app.add_exception_handler(NotADirectoryError, _handle_exception)
    app.add_exception_handler(ValueError, _handle_exception)
    app.add_exception_handler(pymongo_errors.PyMongoError, _handle_exception)
    app.add_exception_handler(QdrantException, _handle_exception)
    app.add_exception_handler(ApiException, _handle_exception)
    app.add_exception_handler(ResponseHandlingException, _handle_exception)
    app.add_exception_handler(UnexpectedResponse, _handle_exception)
    app.add_exception_handler(Exception, _handle_exception)

    @app.middleware("http")
    async def _fallback_error_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - exercised via API tests
            status_code, code, message, details = _map_exception(exc)
            request_id = getattr(request.state, "request_id", None) or get_request_id()
            return build_error_response(
                status_code=status_code,
                code=code,
                message=message,
                details=details,
                request_id=request_id,
            )
