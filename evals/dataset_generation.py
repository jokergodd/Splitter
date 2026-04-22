from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.documents import Document

_TESTSET_GENERATOR_IMPORT_ERROR: ImportError | None = None

try:
    from ragas.testset.synthesizers.generate import TestsetGenerator
except ImportError as exc:  # pragma: no cover - exercised via build_ragas_generator guard
    TestsetGenerator = None
    _TESTSET_GENERATOR_IMPORT_ERROR = exc


def parent_chunk_record_to_document(
    record: Mapping[str, Any],
    *,
    allow_blank_text: bool = False,
) -> Document:
    parent_id = record.get("parent_id")
    if parent_id is None:
        raise ValueError("parent chunk record is missing parent_id")

    text = record.get("text")
    if text is None or (not allow_blank_text and not str(text).strip()):
        raise ValueError("parent chunk record is missing text")

    raw_metadata = record.get("metadata")
    if raw_metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_metadata, Mapping):
        metadata = dict(raw_metadata)
    else:
        raise TypeError("parent chunk record metadata must be a mapping")

    normalized_parent_id = str(parent_id)
    metadata["parent_id"] = normalized_parent_id

    mongo_id = record.get("_id")
    if mongo_id is not None and "mongo_id" not in metadata:
        metadata["mongo_id"] = str(mongo_id)

    return Document(
        page_content=str(text),
        metadata=metadata,
        id=normalized_parent_id,
    )


def parent_chunk_records_to_documents(
    records: Iterable[Mapping[str, Any]],
    *,
    skip_blank_text: bool = True,
) -> list[Document]:
    documents: list[Document] = []
    for record in records:
        text = record.get("text")
        if skip_blank_text and (text is None or not str(text).strip()):
            continue
        documents.append(
            parent_chunk_record_to_document(record, allow_blank_text=not skip_blank_text)
        )
    return documents


def build_ragas_generator(*, llm: Any, embedding_model: Any) -> Any:
    if TestsetGenerator is None:
        raise ImportError("ragas TestsetGenerator is unavailable") from _TESTSET_GENERATOR_IMPORT_ERROR

    from_langchain = getattr(TestsetGenerator, "from_langchain", None)
    if callable(from_langchain):
        return from_langchain(llm, embedding_model)

    return TestsetGenerator(llm=llm, embedding_model=embedding_model)


def _resolve_generate_method(generator: Any) -> Any:
    for method_name in ("generate_with_chunks", "generate_with_langchain_docs", "generate"):
        method = getattr(generator, method_name, None)
        if callable(method):
            return method
    raise TypeError(
        "generator must define generate_with_chunks, generate_with_langchain_docs or generate"
    )


def generate_synthetic_testset(
    records: Iterable[Mapping[str, Any]],
    *,
    testset_size: int,
    generator: Any,
    skip_blank_text: bool = True,
    **generate_kwargs: Any,
) -> Any:
    if testset_size <= 0:
        raise ValueError("testset_size must be positive")

    documents = parent_chunk_records_to_documents(records, skip_blank_text=skip_blank_text)
    generate_method = _resolve_generate_method(generator)
    method_name = getattr(generate_method, "__name__", "")

    if method_name == "generate":
        return generate_method(
            testset_size=testset_size,
            documents=documents,
            **generate_kwargs,
        )
    if method_name == "generate_with_chunks":
        return generate_method(documents, testset_size=testset_size, **generate_kwargs)
    return generate_method(documents, testset_size=testset_size, **generate_kwargs)


__all__ = [
    "TestsetGenerator",
    "build_ragas_generator",
    "generate_synthetic_testset",
    "parent_chunk_record_to_document",
    "parent_chunk_records_to_documents",
]
