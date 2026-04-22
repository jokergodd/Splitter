from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from rag_demo.reranker_runtime import DEFAULT_RERANKER_MODEL

DEFAULT_APP_NAME = "Splitter API"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TASK_MAX_WORKERS = 4
DEFAULT_DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SPARSE_MODEL_NAME = "Qdrant/bm25"


@dataclass(slots=True)
class AppSettings:
    app_name: str = DEFAULT_APP_NAME
    app_version: str = DEFAULT_APP_VERSION
    api_host: str = DEFAULT_API_HOST
    api_port: int = DEFAULT_API_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    task_max_workers: int = DEFAULT_TASK_MAX_WORKERS


@dataclass(slots=True)
class RuntimeSettings:
    dense_model_name: str = DEFAULT_DENSE_MODEL_NAME
    sparse_model_name: str = DEFAULT_SPARSE_MODEL_NAME
    reranker_model_name: str = DEFAULT_RERANKER_MODEL
    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    deepseek_model: str = ""


@dataclass(slots=True)
class Settings:
    app: AppSettings
    runtime: RuntimeSettings


def _load_dotenv() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _required(name: str) -> str:
    value = _get_env(name)
    if not value:
        raise ValueError(f"Missing required setting: {name}")
    return value


def load_settings() -> Settings:
    _load_dotenv()
    app = AppSettings(
        app_name=_get_env("APP_NAME", DEFAULT_APP_NAME) or DEFAULT_APP_NAME,
        app_version=_get_env("APP_VERSION", DEFAULT_APP_VERSION) or DEFAULT_APP_VERSION,
        api_host=_get_env("API_HOST", DEFAULT_API_HOST) or DEFAULT_API_HOST,
        api_port=int(_get_env("API_PORT", str(DEFAULT_API_PORT)) or DEFAULT_API_PORT),
        log_level=(_get_env("LOG_LEVEL", DEFAULT_LOG_LEVEL) or DEFAULT_LOG_LEVEL).upper(),
        task_max_workers=int(
            _get_env("TASK_MAX_WORKERS", str(DEFAULT_TASK_MAX_WORKERS)) or DEFAULT_TASK_MAX_WORKERS
        ),
    )
    runtime = RuntimeSettings(
        dense_model_name=_get_env("DENSE_MODEL_NAME", DEFAULT_DENSE_MODEL_NAME)
        or DEFAULT_DENSE_MODEL_NAME,
        sparse_model_name=_get_env("SPARSE_MODEL_NAME", DEFAULT_SPARSE_MODEL_NAME)
        or DEFAULT_SPARSE_MODEL_NAME,
        reranker_model_name=_get_env("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL)
        or DEFAULT_RERANKER_MODEL,
        deepseek_api_key=_required("DEEPSEEK_API_KEY"),
        deepseek_base_url=_required("DEEPSEEK_BASE_URL"),
        deepseek_model=_required("DEEPSEEK_MODEL"),
    )
    return Settings(app=app, runtime=runtime)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


__all__ = [
    "AppSettings",
    "RuntimeSettings",
    "Settings",
    "DEFAULT_APP_NAME",
    "DEFAULT_APP_VERSION",
    "DEFAULT_API_HOST",
    "DEFAULT_API_PORT",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_TASK_MAX_WORKERS",
    "DEFAULT_DENSE_MODEL_NAME",
    "DEFAULT_SPARSE_MODEL_NAME",
    "get_settings",
    "load_settings",
]
