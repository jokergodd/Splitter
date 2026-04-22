from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any

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


def get_chat_service(runtime: Any = None) -> Any:
    service_cls = _load_service("services.chat_service", "ChatService", ChatService)
    return _instantiate_service(service_cls, runtime or get_runtime())


def get_ingest_service(runtime: Any = None) -> Any:
    service_cls = _load_service("services.ingest_service", "IngestService", IngestService)
    return _instantiate_service(service_cls, runtime or get_runtime())


def get_task_service(runtime: Any = None) -> Any:
    service_cls = _load_service("services.task_service", "TaskService", TaskService)
    active_runtime = runtime or get_runtime()
    settings = get_settings()
    try:
        return service_cls(runtime=active_runtime, max_workers=settings.app.task_max_workers)
    except TypeError:
        return _instantiate_service(service_cls, active_runtime)
