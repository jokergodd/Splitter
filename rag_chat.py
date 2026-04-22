from __future__ import annotations

from dataclasses import dataclass

from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

from rag_demo.answering import AnswerResult, answer_query
from rag_demo.embeddings import CachedEmbeddings
from rag_demo.llm import build_deepseek_llm, load_deepseek_config
from rag_demo.reranker_runtime import build_cross_encoder_reranker
from rag_demo.storage import build_storage_backend

DEFAULT_DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SPARSE_MODEL_NAME = "Qdrant/bm25"


@dataclass(slots=True)
class Runtime:
    llm: object
    dense_embeddings: object
    eval_llm: object | None
    eval_embeddings: object | None
    sparse_embeddings: object
    reranker: object
    storage_backend: object


def build_runtime() -> Runtime:
    config = load_deepseek_config()
    llm = build_deepseek_llm(config)
    dense_embeddings = CachedEmbeddings(
        HuggingFaceEmbeddings(model_name=DEFAULT_DENSE_MODEL_NAME)
    )
    sparse_embeddings = SparseTextEmbedding(model_name=DEFAULT_SPARSE_MODEL_NAME)
    reranker = build_cross_encoder_reranker()
    storage_backend = build_storage_backend(sparse_embeddings=sparse_embeddings)
    return Runtime(
        llm=llm,
        dense_embeddings=dense_embeddings,
        eval_llm=None,
        eval_embeddings=None,
        sparse_embeddings=sparse_embeddings,
        reranker=reranker,
        storage_backend=storage_backend,
    )


def _response_text(response) -> str:
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content
    return str(response)


def _answer_question(question: str, runtime: Runtime) -> AnswerResult:
    hybrid_store = runtime.storage_backend.qdrant_store

    return answer_query(
        original_query=question,
        llm=runtime.llm,
        client=hybrid_store.client,
        collection_name=hybrid_store.collection_name,
        embeddings=runtime.dense_embeddings,
        sparse_embeddings=runtime.sparse_embeddings,
        mongo_repository=runtime.storage_backend.mongo_repository,
        reranker=runtime.reranker,
    )


def _print_sources(source_items: list[dict[str, object]]) -> None:
    print("Sources:")
    if not source_items:
        print("- none")
        return

    for source_item in source_items:
        print(
            "- "
            f"parent_id={source_item.get('parent_id')} "
            f"source={source_item.get('source')} "
            f"file_path={source_item.get('file_path')}"
        )


def main() -> int:
    runtime = build_runtime()

    while True:
        try:
            question = input("Question> ").strip()
        except EOFError:
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            result = _answer_question(question, runtime)
        except Exception as exc:  # pragma: no cover - defensive loop guard
            print(f"Error: {exc}")
            continue

        print("Answer:")
        print(result.answer)
        _print_sources(result.source_items)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
