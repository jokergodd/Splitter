from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from rag_demo.chunking import ChunkingConfig
from rag_demo.pipeline import (
    BatchResult,
    FileProcessingResult,
    PipelineConfig,
    discover_supported_files,
    run_batch_pipeline,
)


class _FakeEmbeddings:
    pass


def test_discover_supported_files_returns_sorted_one_level_supported_types(tmp_path):
    data_dir = tmp_path / "data"
    nested_dir = data_dir / "nested"
    nested_dir.mkdir(parents=True)
    data_dir.mkdir(exist_ok=True)

    second_pdf = data_dir / "b.pdf"
    first_pdf = data_dir / "a.PDF"
    docx_file = data_dir / "c.docx"
    markdown_file = data_dir / "d.md"
    nested_pdf = nested_dir / "z.pdf"
    text_file = data_dir / "notes.txt"
    ignored_file = data_dir / "ignored.csv"

    first_pdf.write_bytes(b"%PDF-1.4\n")
    second_pdf.write_bytes(b"%PDF-1.4\n")
    docx_file.write_bytes(b"docx")
    markdown_file.write_text("# hello", encoding="utf-8")
    nested_pdf.write_bytes(b"%PDF-1.4\n")
    text_file.write_text("plain text", encoding="utf-8")
    ignored_file.write_text("a,b", encoding="utf-8")

    discovered = discover_supported_files(data_dir)

    assert discovered == [first_pdf, second_pdf, docx_file, markdown_file, text_file]


def test_discover_supported_files_rejects_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_dir))):
        discover_supported_files(missing_dir)


def test_discover_supported_files_returns_empty_list_for_directory_without_supported_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.csv").write_text("hello", encoding="utf-8")

    assert discover_supported_files(data_dir) == []


def test_run_batch_pipeline_returns_empty_result_for_directory_without_supported_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.csv").write_text("hello", encoding="utf-8")

    result = run_batch_pipeline(data_dir, embeddings=_FakeEmbeddings())

    assert result.directory == data_dir
    assert result.total_files == 0
    assert result.successful_files == 0
    assert result.skipped_files == 0
    assert result.failed_files == 0
    assert result.files == []


def test_run_batch_pipeline_collects_successful_results(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first_pdf = data_dir / "a.pdf"
    second_docx = data_dir / "b.docx"
    first_pdf.write_bytes(b"%PDF-1.4\n")
    second_docx.write_bytes(b"docx")

    run_calls: list[tuple[object, PipelineConfig]] = []

    def fake_run_document_pipeline(file_path, config, embeddings, storage_backend=None):
        run_calls.append((file_path, config))
        return SimpleNamespace(
            raw_page_count=4,
            cleaned_page_count=4,
            parent_chunks=[object(), object()],
            child_chunks=[object(), object(), object()],
            status="ok",
            content_hash="hash-1",
            skip_reason=None,
        )

    monkeypatch.setattr("rag_demo.pipeline.run_document_pipeline", fake_run_document_pipeline)

    result = run_batch_pipeline(data_dir, embeddings=_FakeEmbeddings())

    assert isinstance(result, BatchResult)
    assert result.directory == data_dir
    assert result.total_files == 2
    assert result.successful_files == 2
    assert result.skipped_files == 0
    assert result.failed_files == 0
    assert [file_result.file_path for file_result in result.files] == [first_pdf, second_docx]
    assert all(isinstance(file_result, FileProcessingResult) for file_result in result.files)
    assert [file_result.status for file_result in result.files] == ["ok", "ok"]
    assert result.files[0].raw_page_count == 4
    assert result.files[0].cleaned_page_count == 4
    assert result.files[0].parent_chunk_count == 2
    assert result.files[0].child_chunk_count == 3
    assert result.files[0].content_hash == "hash-1"
    assert result.files[0].error is None
    assert run_calls == [
        (first_pdf, PipelineConfig()),
        (second_docx, PipelineConfig()),
    ]


def test_run_batch_pipeline_preserves_chunking_config(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    markdown_file = data_dir / "a.md"
    markdown_file.write_text("# heading", encoding="utf-8")

    captured_configs: list[PipelineConfig] = []

    def fake_run_document_pipeline(file_path, config, embeddings, storage_backend=None):
        captured_configs.append(config)
        return SimpleNamespace(
            raw_page_count=1,
            cleaned_page_count=1,
            parent_chunks=[object()],
            child_chunks=[object()],
            status="ok",
            content_hash="hash-1",
            skip_reason=None,
        )

    monkeypatch.setattr("rag_demo.pipeline.run_document_pipeline", fake_run_document_pipeline)

    batch_config = PipelineConfig(
        chunking=ChunkingConfig(
            parent_chunk_size=777,
            parent_chunk_overlap=33,
            child_chunk_size=222,
        )
    )

    result = run_batch_pipeline(data_dir, embeddings=_FakeEmbeddings(), pipeline_config=batch_config)

    assert result.successful_files == 1
    assert captured_configs == [PipelineConfig(chunking=batch_config.chunking)]


def test_run_batch_pipeline_records_file_failure_and_continues(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first_pdf = data_dir / "a.pdf"
    second_txt = data_dir / "b.txt"
    first_pdf.write_bytes(b"%PDF-1.4\n")
    second_txt.write_text("plain text", encoding="utf-8")

    run_calls: list[object] = []

    def fake_run_document_pipeline(file_path, config, embeddings, storage_backend=None):
        run_calls.append(file_path)
        if file_path == first_pdf:
            raise RuntimeError("parse failed")
        return SimpleNamespace(
            raw_page_count=2,
            cleaned_page_count=2,
            parent_chunks=[object()],
            child_chunks=[object()],
            status="ok",
            content_hash="hash-2",
            skip_reason=None,
        )

    monkeypatch.setattr("rag_demo.pipeline.run_document_pipeline", fake_run_document_pipeline)

    result = run_batch_pipeline(data_dir, embeddings=_FakeEmbeddings())

    assert result.total_files == 2
    assert result.successful_files == 1
    assert result.skipped_files == 0
    assert result.failed_files == 1
    assert [file_result.file_path for file_result in result.files] == [first_pdf, second_txt]
    assert result.files[0].status == "failed"
    assert "parse failed" in (result.files[0].error or "")
    assert result.files[1].status == "ok"
    assert result.files[1].raw_page_count == 2
    assert result.files[1].parent_chunk_count == 1
    assert result.files[1].child_chunk_count == 1
    assert run_calls == [first_pdf, second_txt]


def test_run_batch_pipeline_counts_skipped_files(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first_pdf = data_dir / "a.pdf"
    second_txt = data_dir / "b.txt"
    first_pdf.write_bytes(b"%PDF-1.4\n")
    second_txt.write_text("plain text", encoding="utf-8")

    def fake_run_document_pipeline(file_path, config, embeddings, storage_backend=None):
        if file_path == first_pdf:
            return SimpleNamespace(
                raw_page_count=0,
                cleaned_page_count=0,
                parent_chunks=[],
                child_chunks=[],
                status="skipped",
                content_hash="hash-dup",
                skip_reason="content hash already exists",
            )
        return SimpleNamespace(
            raw_page_count=1,
            cleaned_page_count=1,
            parent_chunks=[object()],
            child_chunks=[object()],
            status="ok",
            content_hash="hash-ok",
            skip_reason=None,
        )

    monkeypatch.setattr("rag_demo.pipeline.run_document_pipeline", fake_run_document_pipeline)

    result = run_batch_pipeline(data_dir, embeddings=_FakeEmbeddings())

    assert result.total_files == 2
    assert result.successful_files == 1
    assert result.skipped_files == 1
    assert result.failed_files == 0
    assert result.files[0].status == "skipped"
    assert result.files[0].skip_reason == "content hash already exists"
