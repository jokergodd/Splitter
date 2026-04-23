from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any

from fastapi import Request

from runtime.settings import get_settings as load_settings

runtime_container: Any = None
ChatService: Any = None
IngestService: Any = None
TaskService: Any = None


@lru_cache(maxsize=1)
def get_settings() -> Any:
    return load_settings()


def _load_runtime_container() -> Any:
    if runtime_container is not None:
        return runtime_container
    return import_module("runtime.container")


@lru_cache(maxsize=1)
def get_runtime() -> Any:
    container = _load_runtime_container()
    if hasattr(container, "get_runtime"):
        return container.get_runtime()
    if hasattr(container, "build_runtime"):
        return container.build_runtime()
    raise RuntimeError("runtime.container must define get_runtime() or build_runtime()")


def _load_service(module_name: str, class_name: str, fallback: Any) -> Any:
    if fallback is not None:
        return fallback
    module = import_module(module_name)
    return getattr(module, class_name)


def _instantiate_service(service_cls: Any, runtime: Any) -> Any:
    for factory in (
        lambda: service_cls(runtime=runtime),
        lambda: service_cls(runtime),
        lambda: service_cls(),
    ):
        try:
            return factory()
        except TypeError:
            continue
    raise RuntimeError(f"Unable to instantiate {service_cls!r}")


def get_chat_service(request: Request, runtime: Any = None) -> Any:
    service_cls = _load_service("services.chat_service", "ChatService", ChatService)
    active_runtime = runtime
    if active_runtime is None and request is not None and hasattr(request.app.state, "runtime"):
        active_runtime = request.app.state.runtime
        if active_runtime is None:
            active_runtime = get_runtime()
            request.app.state.runtime = active_runtime
    return _instantiate_service(service_cls, active_runtime or get_runtime())


def get_ingest_runtime(request: Request) -> Any:
    container = _load_runtime_container()
    if request is not None and hasattr(request.app.state, "ingest_runtime"):
        active_runtime = request.app.state.ingest_runtime
        if active_runtime is None:
            active_runtime = container.get_ingest_runtime()
            request.app.state.ingest_runtime = active_runtime
        return active_runtime
    return container.get_ingest_runtime()


def get_ingest_service(request: Request, runtime: Any = None) -> Any:
    service_cls = _load_service("services.ingest_service", "IngestService", IngestService)
    active_runtime = runtime or get_ingest_runtime(request)
    return _instantiate_service(service_cls, active_runtime)


@lru_cache(maxsize=1)
def _get_default_task_service() -> Any:
    service_cls = _load_service("services.task_service", "TaskService", TaskService)
    settings = get_settings()
    active_runtime = _load_runtime_container().get_ingest_runtime()
    try:
        return service_cls(runtime=active_runtime, max_workers=settings.app.task_max_workers)
    except TypeError:
        return _instantiate_service(service_cls, active_runtime)


def get_task_service(request: Request, runtime: Any = None) -> Any:
    if request is not None and hasattr(request.app.state, "task_service"):
        task_service = request.app.state.task_service
        if task_service is None:
            service_cls = _load_service("services.task_service", "TaskService", TaskService)
            settings = get_settings()
            active_runtime = runtime or get_ingest_runtime(request)
            try:
                task_service = service_cls(runtime=active_runtime, max_workers=settings.app.task_max_workers)
            except TypeError:
                task_service = _instantiate_service(service_cls, active_runtime)
            request.app.state.task_service = task_service
        return task_service
    if runtime is not None:
        service_cls = _load_service("services.task_service", "TaskService", TaskService)
        settings = get_settings()
        try:
            return service_cls(runtime=runtime, max_workers=settings.app.task_max_workers)
        except TypeError:
            return _instantiate_service(service_cls, runtime)
    return _get_default_task_service()
