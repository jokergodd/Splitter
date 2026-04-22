from __future__ import annotations

import asyncio

from rag_demo.answering import AnswerResult
from runtime.container import (
    DEFAULT_DENSE_MODEL_NAME,
    DEFAULT_SPARSE_MODEL_NAME,
    Runtime,
    build_runtime as build_shared_runtime,
)
from rag_demo.embeddings import CachedEmbeddings
from rag_demo.llm import build_deepseek_llm, load_deepseek_config
from rag_demo.reranker_runtime import build_cross_encoder_reranker
from rag_demo.storage import build_storage_backend
from services.chat_service import ChatService
from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings


def build_runtime() -> Runtime:
    return build_shared_runtime(
        load_config=load_deepseek_config,
        build_llm=build_deepseek_llm,
        dense_embeddings_factory=HuggingFaceEmbeddings,
        cached_embeddings_factory=CachedEmbeddings,
        sparse_embeddings_factory=SparseTextEmbedding,
        build_reranker=build_cross_encoder_reranker,
        build_storage=build_storage_backend,
        dense_model_name=DEFAULT_DENSE_MODEL_NAME,
        sparse_model_name=DEFAULT_SPARSE_MODEL_NAME,
    )


def _response_text(response) -> str:
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content
    return str(response)


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
    chat_service = ChatService(runtime)

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
            result = asyncio.run(chat_service.answer(question=question))
        except Exception as exc:  # pragma: no cover - defensive loop guard
            print(f"Error: {exc}")
            continue

        print("Answer:")
        print(result.answer)
        _print_sources(result.source_items)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
