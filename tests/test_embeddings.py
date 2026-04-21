from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag_demo.embeddings import CachedEmbeddings


class _FakeEmbeddings:
    def __init__(self):
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [float(len(text))]


def test_cached_embeddings_reuses_document_vectors_for_repeated_text():
    base = _FakeEmbeddings()
    cached = CachedEmbeddings(base)

    first = cached.embed_documents(["alpha", "beta", "alpha"])
    second = cached.embed_documents(["beta", "alpha"])

    assert first == [[5.0], [4.0], [5.0]]
    assert second == [[4.0], [5.0]]
    assert base.document_calls == [["alpha", "beta"]]


def test_cached_embeddings_preserves_input_order_for_mixed_cache_hits():
    base = _FakeEmbeddings()
    cached = CachedEmbeddings(base)

    cached.embed_documents(["alpha"])
    result = cached.embed_documents(["gamma", "alpha", "beta"])

    assert result == [[5.0], [5.0], [4.0]]
    assert base.document_calls == [["alpha"], ["gamma", "beta"]]


def test_cached_embeddings_reuses_query_vectors():
    base = _FakeEmbeddings()
    cached = CachedEmbeddings(base)

    first = cached.embed_query("question")
    second = cached.embed_query("question")

    assert first == [8.0]
    assert second == [8.0]
    assert base.query_calls == ["question"]


def test_cached_embeddings_handles_one_shot_iterables():
    base = _FakeEmbeddings()
    cached = CachedEmbeddings(base)

    texts = (text for text in ["alpha", "beta", "alpha"])

    result = cached.embed_documents(texts)

    assert result == [[5.0], [4.0], [5.0]]
    assert base.document_calls == [["alpha", "beta"]]


def test_cached_embeddings_is_langchain_embedding_and_callable():
    base = _FakeEmbeddings()
    cached = CachedEmbeddings(base)

    assert isinstance(cached, Embeddings)
    assert callable(cached)
    assert cached(["alpha", "beta"]) == [[5.0], [4.0]]
