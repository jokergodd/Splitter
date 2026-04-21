from __future__ import annotations

from langchain_core.embeddings import Embeddings


class CachedEmbeddings(Embeddings):
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings
        self._document_cache: dict[str, list[float]] = {}
        self._query_cache: dict[str, list[float]] = {}

    def __call__(self, texts):
        return self.embed_documents(texts)

    def embed_documents(self, texts):
        texts = list(texts)
        cached_vectors = []
        missing_texts = []

        for text in texts:
            if text in self._document_cache:
                cached_vectors.append(self._document_cache[text])
                continue

            cached_vectors.append(None)
            if text not in missing_texts:
                missing_texts.append(text)

        if missing_texts:
            missing_vectors = self.base_embeddings.embed_documents(missing_texts)
            for text, vector in zip(missing_texts, missing_vectors):
                self._document_cache[text] = vector

        for index, text in enumerate(texts):
            if cached_vectors[index] is None:
                cached_vectors[index] = self._document_cache[text]

        return cached_vectors

    def embed_query(self, text):
        if text not in self._query_cache:
            self._query_cache[text] = self.base_embeddings.embed_query(text)
        return self._query_cache[text]
