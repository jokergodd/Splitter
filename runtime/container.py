from __future__ import annotations

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
