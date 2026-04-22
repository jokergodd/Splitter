from __future__ import annotations

from typing import Any


class DomainError(Exception):
    status_code = 400
    code = "DOMAIN_ERROR"
    default_message = "Domain error"

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None) -> None:
        self.message = message or self.default_message
        self.details = {} if details is None else details
        super().__init__(self.message)


class InvalidRequestError(DomainError):
    status_code = 400
    code = "BAD_REQUEST"
    default_message = "Bad request"


class UnsupportedFileTypeError(DomainError):
    status_code = 400
    code = "UNSUPPORTED_FILE_TYPE"
    default_message = "Unsupported file type"

    def __init__(self, file_type: str, *, supported_types: list[str] | None = None) -> None:
        self.file_type = file_type
        self.supported_types = supported_types or []
        details: dict[str, Any] = {"file_type": file_type}
        if supported_types:
            details["supported_types"] = supported_types
        super().__init__(details=details)


class CollectionNotReadyError(DomainError):
    status_code = 503
    code = "COLLECTION_NOT_READY"
    default_message = "Collection is not ready"

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        super().__init__(message=f"Collection '{collection_name}' is not ready")


class NoContextRetrievedError(DomainError):
    status_code = 404
    code = "NO_CONTEXT_RETRIEVED"
    default_message = "No relevant context was retrieved"

    def __init__(self, query: str | None = None) -> None:
        self.question = query
        super().__init__()


class IngestConflictError(DomainError):
    status_code = 409
    code = "INGEST_CONFLICT"
    default_message = "Ingest request conflicts with existing state"

    def __init__(self, *, content_hash: str | None = None, reason: str | None = None) -> None:
        self.content_hash = content_hash
        self.reason = reason
        details: dict[str, Any] = {}
        if content_hash is not None:
            details["content_hash"] = content_hash
        if reason is not None:
            details["reason"] = reason
        super().__init__(details=details)


class ModelInitializationError(DomainError):
    status_code = 503
    code = "MODEL_INITIALIZATION_ERROR"
    default_message = "Model initialization failed"

    def __init__(self, component: str | None = None) -> None:
        self.component = component
        details: dict[str, Any] = {}
        if component is not None:
            details["component"] = component
        super().__init__(details=details)


class DependencyUnavailableError(DomainError):
    status_code = 503
    code = "DEPENDENCY_UNAVAILABLE"
    default_message = "Required dependency is unavailable"

    def __init__(self, message: str | None = None, *, dependency: str | None = None) -> None:
        self.dependency = dependency
        details: dict[str, Any] = {}
        if dependency is not None:
            details["dependency"] = dependency
        super().__init__(message=message, details=details)


class TaskNotFoundError(DomainError):
    status_code = 404
    code = "TASK_NOT_FOUND"
    default_message = "Task not found"

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__()


__all__ = [
    "CollectionNotReadyError",
    "DependencyUnavailableError",
    "DomainError",
    "InvalidRequestError",
    "IngestConflictError",
    "ModelInitializationError",
    "NoContextRetrievedError",
    "TaskNotFoundError",
    "UnsupportedFileTypeError",
]
