from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import evals.dataset_generation as dataset_generation
from evals.dataset_generation import (
    build_ragas_generator,
    generate_synthetic_testset,
    parent_chunk_record_to_document,
    parent_chunk_records_to_documents,
)


def test_parent_chunk_record_to_document_normalizes_core_fields():
    record = {
        "parent_id": 123,
        "text": "parent chunk body",
        "metadata": {"source": "demo.pdf", "page": 7},
        "_id": "mongo-id-1",
    }

    document = parent_chunk_record_to_document(record)

    assert document == Document(
        page_content="parent chunk body",
        metadata={
            "source": "demo.pdf",
            "page": 7,
            "parent_id": "123",
            "mongo_id": "mongo-id-1",
        },
        id="123",
    )


def test_parent_chunk_record_to_document_rejects_missing_text():
    with pytest.raises(ValueError, match="text"):
        parent_chunk_record_to_document({"parent_id": "parent-1", "metadata": {}})


def test_parent_chunk_record_to_document_rejects_missing_parent_id():
    with pytest.raises(ValueError, match="parent_id"):
        parent_chunk_record_to_document({"text": "parent chunk body", "metadata": {}})


def test_parent_chunk_record_to_document_rejects_non_mapping_metadata():
    with pytest.raises(TypeError, match="mapping"):
        parent_chunk_record_to_document(
            {"parent_id": "parent-1", "text": "parent chunk body", "metadata": ["bad"]}
        )


def test_parent_chunk_records_to_documents_skips_blank_text_records():
    records = [
        {"parent_id": "parent-1", "text": "first", "metadata": {"rank": 1}},
        {"parent_id": "parent-2", "text": "   ", "metadata": {"rank": 2}},
        {"parent_id": "parent-3", "text": "third", "metadata": {}},
    ]

    documents = parent_chunk_records_to_documents(records, skip_blank_text=True)

    assert documents == [
        Document(
            page_content="first",
            metadata={"rank": 1, "parent_id": "parent-1"},
            id="parent-1",
        ),
        Document(
            page_content="third",
            metadata={"parent_id": "parent-3"},
            id="parent-3",
        ),
    ]


def test_parent_chunk_records_to_documents_keeps_blank_text_when_requested():
    records = [
        {"parent_id": "parent-1", "text": "   ", "metadata": {"rank": 1}},
    ]

    documents = parent_chunk_records_to_documents(records, skip_blank_text=False)

    assert documents == [
        Document(
            page_content="   ",
            metadata={"rank": 1, "parent_id": "parent-1"},
            id="parent-1",
        )
    ]


def test_generate_synthetic_testset_converts_records_and_calls_generator():
    calls: list[dict[str, object]] = []

    class _FakeGenerator:
        def generate_with_langchain_docs(self, documents, testset_size, **kwargs):
            calls.append(
                {
                    "documents": list(documents),
                    "testset_size": testset_size,
                    "kwargs": kwargs,
                }
            )
            return {"ok": True}

    records = [
        {"parent_id": "parent-1", "text": "first parent", "metadata": {"source": "a.md"}},
        {"parent_id": "parent-2", "text": "second parent", "metadata": {"source": "b.md"}},
    ]

    result = generate_synthetic_testset(
        records,
        testset_size=5,
        generator=_FakeGenerator(),
        transforms="custom-transforms",
        query_distribution="custom-distribution",
        raise_exceptions=False,
    )

    assert result == {"ok": True}
    assert calls == [
        {
            "documents": [
                Document(
                    page_content="first parent",
                    metadata={"source": "a.md", "parent_id": "parent-1"},
                    id="parent-1",
                ),
                Document(
                    page_content="second parent",
                    metadata={"source": "b.md", "parent_id": "parent-2"},
                    id="parent-2",
                ),
            ],
            "testset_size": 5,
            "kwargs": {
                "transforms": "custom-transforms",
                "query_distribution": "custom-distribution",
                "raise_exceptions": False,
            },
        }
    ]


def test_generate_synthetic_testset_prefers_prechunked_generator_entrypoint():
    calls: list[dict[str, object]] = []

    class _FakeGenerator:
        def generate_with_chunks(self, chunks, testset_size, **kwargs):
            calls.append(
                {
                    "chunks": list(chunks),
                    "testset_size": testset_size,
                    "kwargs": kwargs,
                }
            )
            return {"ok": "chunks"}

        def generate_with_langchain_docs(self, documents, testset_size, **kwargs):
            raise AssertionError("generate_with_langchain_docs should not be used")

    result = generate_synthetic_testset(
        [{"parent_id": "parent-1", "text": "first parent", "metadata": {"source": "a.md"}}],
        testset_size=3,
        generator=_FakeGenerator(),
        raise_exceptions=False,
    )

    assert result == {"ok": "chunks"}
    assert calls == [
        {
            "chunks": [
                Document(
                    page_content="first parent",
                    metadata={"source": "a.md", "parent_id": "parent-1"},
                    id="parent-1",
                )
            ],
            "testset_size": 3,
            "kwargs": {
                "raise_exceptions": False,
            },
        }
    ]


def test_generate_synthetic_testset_falls_back_to_generate_method():
    calls: list[dict[str, object]] = []

    class _CompatGenerator:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return {"ok": "compat"}

    result = generate_synthetic_testset(
        [{"parent_id": "parent-1", "text": "first parent", "metadata": {}}],
        testset_size=2,
        generator=_CompatGenerator(),
        query_distribution="custom-distribution",
    )

    assert result == {"ok": "compat"}
    assert calls == [
        {
            "testset_size": 2,
            "documents": [
                Document(
                    page_content="first parent",
                    metadata={"parent_id": "parent-1"},
                    id="parent-1",
                )
            ],
            "query_distribution": "custom-distribution",
        }
    ]


def test_generate_synthetic_testset_rejects_non_positive_testset_size():
    with pytest.raises(ValueError, match="testset_size"):
        generate_synthetic_testset([], testset_size=0, generator=SimpleNamespace())


def test_generate_synthetic_testset_rejects_generator_without_supported_entrypoint():
    with pytest.raises(TypeError, match="generate_with_chunks|generate_with_langchain_docs|generate"):
        generate_synthetic_testset(
            [{"parent_id": "parent-1", "text": "first", "metadata": {}}],
            testset_size=1,
            generator=SimpleNamespace(),
        )


def test_build_ragas_generator_uses_ragas_factory(monkeypatch):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    fake_factory = SimpleNamespace(
        from_langchain=lambda *args, **kwargs: calls.append((args, kwargs))
        or "generator"
    )

    monkeypatch.setattr(dataset_generation, "TestsetGenerator", fake_factory)

    llm = object()
    embedding_model = object()

    generator = build_ragas_generator(llm=llm, embedding_model=embedding_model)

    assert generator == "generator"
    assert calls == [((llm, embedding_model), {})]


def test_build_ragas_generator_preserves_import_error_cause(monkeypatch):
    original = ImportError("missing ragas dependency")

    monkeypatch.setattr(dataset_generation, "_TESTSET_GENERATOR_IMPORT_ERROR", original)
    monkeypatch.setattr(dataset_generation, "TestsetGenerator", None)

    with pytest.raises(ImportError, match="TestsetGenerator") as exc_info:
        build_ragas_generator(llm=object(), embedding_model=object())

    assert exc_info.value.__cause__ is original
