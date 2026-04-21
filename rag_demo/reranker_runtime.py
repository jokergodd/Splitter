from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _document_text(document: str | Document) -> str:
    if isinstance(document, Document):
        return document.page_content
    return document


class CrossEncoderScorer:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score_batch(self, query: str, documents: Sequence[str | Document]) -> list[float]:
        pairs = [(query, _document_text(document)) for document in documents]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]

    def score(self, query: str, document: str | Document) -> float:
        return self.score_batch(query, [document])[0]

    def __call__(self, query: str, document: str | Document) -> float:
        return self.score(query, document)


def build_cross_encoder_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoderScorer:
    return CrossEncoderScorer(model_name=model_name)


__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "CrossEncoderScorer",
    "build_cross_encoder_reranker",
]
