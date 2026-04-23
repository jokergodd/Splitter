from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

from rag_demo.embeddings import CachedEmbeddings
from rag_demo.llm import build_deepseek_llm, load_deepseek_config
from rag_demo.reranker_runtime import build_cross_encoder_reranker
from rag_demo.storage import build_storage_backend
from runtime.settings import DEFAULT_DENSE_MODEL_NAME, DEFAULT_SPARSE_MODEL_NAME, Settings, get_settings


@dataclass(slots=True)
class Runtime:
    llm: Any
    dense_embeddings: Any
    eval_llm: Any | None
    eval_embeddings: Any | None
    sparse_embeddings: Any
    reranker: Any
    storage_backend: Any


async def close_runtime(runtime: Runtime | None) -> None:
    if runtime is None:
        return

    storage_backend = getattr(runtime, "storage_backend", None)
    mongo_repository = getattr(storage_backend, "mongo_repository", None)
    qdrant_store = getattr(storage_backend, "qdrant_store", None)

    async_mongo_client = getattr(mongo_repository, "async_client", None)
    if async_mongo_client is not None:
        await async_mongo_client.close()

    async_qdrant_client = getattr(qdrant_store, "async_client", None)
    if async_qdrant_client is not None:
        await async_qdrant_client.close()

    qdrant_client = getattr(qdrant_store, "client", None)
    if qdrant_client is not None and hasattr(qdrant_client, "close"):
        close = qdrant_client.close
        if asyncio.iscoroutinefunction(close):
            await close()
        else:
            await asyncio.to_thread(close)


def _build_embeddings_and_storage(
    *,
    dense_embeddings_factory: Callable[..., Any],
    cached_embeddings_factory: Callable[[Any], Any],
    sparse_embeddings_factory: Callable[..., Any],
    build_storage: Callable[..., Any],
    dense_model_name: str,
    sparse_model_name: str,
) -> tuple[Any, Any, Any]:
    dense_embeddings = cached_embeddings_factory(
        dense_embeddings_factory(model_name=dense_model_name)
    )
    sparse_embeddings = sparse_embeddings_factory(model_name=sparse_model_name)
    storage_backend = build_storage(sparse_embeddings=sparse_embeddings)
    return dense_embeddings, sparse_embeddings, storage_backend


def build_runtime(
    *,
    settings: Settings | None = None,
    load_config: Callable[[], Any] | None = None,
    build_llm: Callable[[Any], Any] | None = None,
    dense_embeddings_factory: Callable[..., Any] | None = None,
    cached_embeddings_factory: Callable[[Any], Any] | None = None,
    sparse_embeddings_factory: Callable[..., Any] | None = None,
    build_reranker: Callable[[], Any] | None = None,
    build_storage: Callable[..., Any] | None = None,
    dense_model_name: str = DEFAULT_DENSE_MODEL_NAME,
    sparse_model_name: str = DEFAULT_SPARSE_MODEL_NAME,
) -> Runtime:
    settings = get_settings() if settings is None else settings
    load_config = load_deepseek_config if load_config is None else load_config
    build_llm = build_deepseek_llm if build_llm is None else build_llm
    dense_embeddings_factory = (
        HuggingFaceEmbeddings if dense_embeddings_factory is None else dense_embeddings_factory
    )
    cached_embeddings_factory = CachedEmbeddings if cached_embeddings_factory is None else cached_embeddings_factory
    sparse_embeddings_factory = SparseTextEmbedding if sparse_embeddings_factory is None else sparse_embeddings_factory
    build_reranker = build_cross_encoder_reranker if build_reranker is None else build_reranker
    build_storage = build_storage_backend if build_storage is None else build_storage

    dense_model_name = settings.runtime.dense_model_name if settings is not None else dense_model_name
    sparse_model_name = settings.runtime.sparse_model_name if settings is not None else sparse_model_name
    config = load_config()
    llm = build_llm(config)
    dense_embeddings, sparse_embeddings, storage_backend = _build_embeddings_and_storage(
        dense_embeddings_factory=dense_embeddings_factory,
        cached_embeddings_factory=cached_embeddings_factory,
        sparse_embeddings_factory=sparse_embeddings_factory,
        build_storage=build_storage,
        dense_model_name=dense_model_name,
        sparse_model_name=sparse_model_name,
    )
    try:
        reranker = build_reranker(settings.runtime.reranker_model_name)
    except TypeError:
        reranker = build_reranker()
    return Runtime(
        llm=llm,
        dense_embeddings=dense_embeddings,
        eval_llm=None,
        eval_embeddings=None,
        sparse_embeddings=sparse_embeddings,
        reranker=reranker,
        storage_backend=storage_backend,
    )


def build_ingest_runtime(
    *,
    settings: Settings | None = None,
    dense_embeddings_factory: Callable[..., Any] | None = None,
    cached_embeddings_factory: Callable[[Any], Any] | None = None,
    sparse_embeddings_factory: Callable[..., Any] | None = None,
    build_storage: Callable[..., Any] | None = None,
    dense_model_name: str = DEFAULT_DENSE_MODEL_NAME,
    sparse_model_name: str = DEFAULT_SPARSE_MODEL_NAME,
) -> Runtime:
    settings = get_settings() if settings is None else settings
    dense_embeddings_factory = (
        HuggingFaceEmbeddings if dense_embeddings_factory is None else dense_embeddings_factory
    )
    cached_embeddings_factory = CachedEmbeddings if cached_embeddings_factory is None else cached_embeddings_factory
    sparse_embeddings_factory = SparseTextEmbedding if sparse_embeddings_factory is None else sparse_embeddings_factory
    build_storage = build_storage_backend if build_storage is None else build_storage

    dense_model_name = settings.runtime.dense_model_name if settings is not None else dense_model_name
    sparse_model_name = settings.runtime.sparse_model_name if settings is not None else sparse_model_name
    dense_embeddings, sparse_embeddings, storage_backend = _build_embeddings_and_storage(
        dense_embeddings_factory=dense_embeddings_factory,
        cached_embeddings_factory=cached_embeddings_factory,
        sparse_embeddings_factory=sparse_embeddings_factory,
        build_storage=build_storage,
        dense_model_name=dense_model_name,
        sparse_model_name=sparse_model_name,
    )
    return Runtime(
        llm=None,
        dense_embeddings=dense_embeddings,
        eval_llm=None,
        eval_embeddings=None,
        sparse_embeddings=sparse_embeddings,
        reranker=None,
        storage_backend=storage_backend,
    )


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return build_runtime(settings=get_settings())


@lru_cache(maxsize=1)
def get_ingest_runtime() -> Runtime:
    return build_ingest_runtime(settings=get_settings())


def clear_runtime_caches() -> None:
    get_runtime.cache_clear()
    get_ingest_runtime.cache_clear()
