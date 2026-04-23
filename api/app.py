from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request

from api import dependencies as api_dependencies
from api.errors import register_exception_handlers
from api.logging import REQUEST_ID_HEADER, generate_request_id, get_logger, reset_request_id, set_request_id
from api.routers.chat import router as chat_router
from api.routers.health import router as health_router
from api.routers.ingest import router as ingest_router
from api.routers.tasks import router as tasks_router
from runtime.container import clear_runtime_caches, close_runtime
from runtime.settings import Settings, get_settings

logger = get_logger(__name__)


async def _request_logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get(REQUEST_ID_HEADER) or generate_request_id()
    token = set_request_id(request_id)
    request.state.request_id = request_id
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        reset_request_id(token)
        raise
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "http_request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    reset_request_id(token)
    return response


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.runtime = None
    app.state.ingest_runtime = None
    app.state.task_service = None
    logger.info(
        "app.startup",
        extra={
            "app_name": app.title,
            "app_version": app.version,
        },
    )
    try:
        yield
    finally:
        task_service = getattr(app.state, "task_service", None)
        if task_service is not None and hasattr(task_service, "close"):
            task_service.close()

        ingest_runtime = getattr(app.state, "ingest_runtime", None)
        runtime = getattr(app.state, "runtime", None)
        if ingest_runtime is not None:
            await close_runtime(ingest_runtime)
        if runtime is not None and runtime is not ingest_runtime:
            await close_runtime(runtime)

        api_dependencies.get_runtime.cache_clear()
        api_dependencies.get_settings.cache_clear()
        api_dependencies._get_default_task_service.cache_clear()
        clear_runtime_caches()
        logger.info(
            "app.shutdown",
            extra={
                "app_name": app.title,
                "app_version": app.version,
            },
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = get_settings() if settings is None else settings
    app = FastAPI(
        title=settings.app.app_name,
        version=settings.app.app_version,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    register_exception_handlers(app)
    app.middleware("http")(_request_logging_middleware)
    app.include_router(chat_router)
    app.include_router(ingest_router)
    app.include_router(tasks_router)
    app.include_router(health_router)
    return app


app = create_app()
