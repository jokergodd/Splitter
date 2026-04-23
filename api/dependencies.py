from __future__ import annotations

import inspect
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
    signature = inspect.signature(service_cls)
    for args, kwargs in (
        ((), {"runtime": runtime}),
        ((runtime,), {}),
        ((), {}),
    ):
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return service_cls(*args, **kwargs)
    raise RuntimeError(f"Unable to instantiate {service_cls!r}")


def _instantiate_chat_service(
    service_cls: Any,
    *,
    runtime: Any | None = None,
    runtime_factory: Any | None = None,
) -> Any:
    signature = inspect.signature(service_cls)
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    if runtime is not None:
        candidates.extend(
            [
                ((), {"runtime": runtime}),
                ((runtime,), {}),
            ]
        )
    if runtime_factory is not None:
        candidates.append(((), {"runtime_factory": runtime_factory}))
    candidates.append(((), {}))

    for call_args, call_kwargs in candidates:
        try:
            signature.bind(*call_args, **call_kwargs)
        except TypeError:
            continue
        return service_cls(*call_args, **call_kwargs)
    raise RuntimeError(f"Unable to instantiate {service_cls!r}")


def get_chat_service(request: Request, runtime: Any = None) -> Any:
    service_cls = _load_service("services.chat_service", "ChatService", ChatService)
    if request is not None and hasattr(request.app.state, "runtime"):
        cached_service = getattr(request.app.state, "chat_service", None)
        active_runtime = runtime if runtime is not None else request.app.state.runtime
        if cached_service is not None and (active_runtime is None or getattr(cached_service, "runtime", None) is active_runtime):
            return cached_service

        def resolve_runtime() -> Any:
            current_runtime = request.app.state.runtime
            if current_runtime is None:
                current_runtime = runtime if runtime is not None else get_runtime()
                request.app.state.runtime = current_runtime
            return current_runtime

        service = _instantiate_chat_service(
            service_cls,
            runtime=active_runtime,
            runtime_factory=None if active_runtime is not None else resolve_runtime,
        )
        request.app.state.chat_service = service
        return service
    return _instantiate_chat_service(service_cls, runtime=runtime or get_runtime())


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
