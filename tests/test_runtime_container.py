from __future__ import annotations

from types import SimpleNamespace

import runtime.container as runtime_container


def test_build_runtime_initializes_shared_dependencies_once(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDenseEmbeddings:
        def __init__(self, *, model_name: str):
            captured["dense_model_name"] = model_name

    class FakeCachedEmbeddings:
        def __init__(self, base_embeddings):
            captured["dense_base_embeddings"] = base_embeddings
            self.base_embeddings = base_embeddings

    class FakeSparseEmbeddings:
        def __init__(self, *, model_name: str):
            captured["sparse_model_name"] = model_name

    fake_llm = object()
    fake_reranker = object()
    fake_storage_backend = object()

    monkeypatch.setattr(runtime_container, "load_deepseek_config", lambda: "config")
    monkeypatch.setattr(runtime_container, "build_deepseek_llm", lambda config: fake_llm)
    monkeypatch.setattr(runtime_container, "HuggingFaceEmbeddings", FakeDenseEmbeddings)
    monkeypatch.setattr(runtime_container, "CachedEmbeddings", FakeCachedEmbeddings)
    monkeypatch.setattr(runtime_container, "SparseTextEmbedding", FakeSparseEmbeddings)
    monkeypatch.setattr(runtime_container, "build_cross_encoder_reranker", lambda: fake_reranker)
    monkeypatch.setattr(
        runtime_container,
        "build_storage_backend",
        lambda *, sparse_embeddings: captured.__setitem__("storage_sparse_embeddings", sparse_embeddings)
        or fake_storage_backend,
    )

    runtime = runtime_container.build_runtime()

    assert runtime.llm is fake_llm
    assert isinstance(runtime.dense_embeddings, FakeCachedEmbeddings)
    assert runtime.dense_embeddings.base_embeddings.__class__ is FakeDenseEmbeddings
    assert runtime.eval_llm is None
    assert runtime.eval_embeddings is None
    assert isinstance(runtime.sparse_embeddings, FakeSparseEmbeddings)
    assert runtime.reranker is fake_reranker
    assert runtime.storage_backend is fake_storage_backend
    assert captured == {
        "dense_model_name": runtime_container.DEFAULT_DENSE_MODEL_NAME,
        "dense_base_embeddings": runtime.dense_embeddings.base_embeddings,
        "sparse_model_name": runtime_container.DEFAULT_SPARSE_MODEL_NAME,
        "storage_sparse_embeddings": runtime.sparse_embeddings,
    }


def test_runtime_package_reexports_container_api():
    import runtime

    assert runtime.Runtime is runtime_container.Runtime
    assert runtime.build_ingest_runtime is runtime_container.build_ingest_runtime
    assert runtime.build_runtime is runtime_container.build_runtime
    assert runtime.get_ingest_runtime is runtime_container.get_ingest_runtime
    assert runtime.get_runtime is runtime_container.get_runtime


def test_build_ingest_runtime_initializes_only_ingest_dependencies(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDenseEmbeddings:
        def __init__(self, *, model_name: str):
            captured["dense_model_name"] = model_name

    class FakeCachedEmbeddings:
        def __init__(self, base_embeddings):
            captured["dense_base_embeddings"] = base_embeddings
            self.base_embeddings = base_embeddings

    class FakeSparseEmbeddings:
        def __init__(self, *, model_name: str):
            captured["sparse_model_name"] = model_name

    fake_storage_backend = object()

    runtime = runtime_container.build_ingest_runtime(
        dense_embeddings_factory=FakeDenseEmbeddings,
        cached_embeddings_factory=FakeCachedEmbeddings,
        sparse_embeddings_factory=FakeSparseEmbeddings,
        build_storage=lambda *, sparse_embeddings: captured.__setitem__(
            "storage_sparse_embeddings", sparse_embeddings
        )
        or fake_storage_backend,
    )

    assert runtime.llm is None
    assert runtime.reranker is None
    assert isinstance(runtime.dense_embeddings, FakeCachedEmbeddings)
    assert runtime.dense_embeddings.base_embeddings.__class__ is FakeDenseEmbeddings
    assert runtime.eval_llm is None
    assert runtime.eval_embeddings is None
    assert isinstance(runtime.sparse_embeddings, FakeSparseEmbeddings)
    assert runtime.storage_backend is fake_storage_backend
    assert captured == {
        "dense_model_name": runtime_container.DEFAULT_DENSE_MODEL_NAME,
        "dense_base_embeddings": runtime.dense_embeddings.base_embeddings,
        "sparse_model_name": runtime_container.DEFAULT_SPARSE_MODEL_NAME,
        "storage_sparse_embeddings": runtime.sparse_embeddings,
    }
