from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pymongo.errors import ServerSelectionTimeoutError
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_demo.pipeline import BatchResult, PipelineConfig, PipelineResult
from services import ingest_service
from services.errors import CollectionNotReadyError, DependencyUnavailableError


def test_ingest_file_uses_async_pipeline_and_forwards_runtime_dependencies(monkeypatch, tmp_path):
    runtime = SimpleNamespace(dense_embeddings=object(), storage_backend=object())
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    config = PipelineConfig()
    expected = PipelineResult(raw_page_count=1, cleaned_page_count=1, parent_chunks=[], child_chunks=[])
    captured: dict[str, object] = {}

    async def fake_run_document_pipeline_async(*, file_path, config, embeddings, storage_backend=None):
        captured["kwargs"] = {
            "file_path": file_path,
            "config": config,
            "embeddings": embeddings,
            "storage_backend": storage_backend,
        }
        return expected

    monkeypatch.setattr(ingest_service, "run_document_pipeline_async", fake_run_document_pipeline_async)

    async def run() -> PipelineResult:
        return await ingest_service.ingest_file(file_path, runtime, config=config)

    result = asyncio.run(run())

    assert result is expected
    assert captured["kwargs"]["file_path"] == file_path
    assert captured["kwargs"]["config"] is config
    assert captured["kwargs"]["embeddings"] is runtime.dense_embeddings
    assert captured["kwargs"]["storage_backend"] is runtime.storage_backend


def test_ingest_batch_uses_async_pipeline_and_default_config(monkeypatch, tmp_path):
    runtime = SimpleNamespace(dense_embeddings=object(), storage_backend=object())
    directory = tmp_path / "batch"
    directory.mkdir()
    expected = BatchResult(directory=directory, total_files=0, successful_files=0, skipped_files=0, failed_files=0, files=[])
    captured: dict[str, object] = {}

    async def fake_run_batch_pipeline_async(*, directory, embeddings, pipeline_config, storage_backend=None):
        captured["kwargs"] = {
            "directory": directory,
            "embeddings": embeddings,
            "pipeline_config": pipeline_config,
            "storage_backend": storage_backend,
        }
        return expected

    monkeypatch.setattr(ingest_service, "run_batch_pipeline_async", fake_run_batch_pipeline_async)

    async def run() -> BatchResult:
        return await ingest_service.ingest_batch(directory, runtime)

    result = asyncio.run(run())

    assert result is expected
    assert captured["kwargs"]["directory"] == directory
    assert captured["kwargs"]["embeddings"] is runtime.dense_embeddings
    assert isinstance(captured["kwargs"]["pipeline_config"], PipelineConfig)
    assert captured["kwargs"]["storage_backend"] is runtime.storage_backend


def test_ingest_service_class_delegates_to_module_functions(monkeypatch, tmp_path):
    runtime = SimpleNamespace()
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    data_dir = tmp_path / "batch"
    data_dir.mkdir()
    expected_file = PipelineResult(raw_page_count=1, cleaned_page_count=1, parent_chunks=[], child_chunks=[])
    expected_batch = BatchResult(
        directory=data_dir,
        total_files=0,
        successful_files=0,
        skipped_files=0,
        failed_files=0,
        files=[],
    )
    calls: list[tuple[str, object, object, object]] = []

    async def fake_ingest_file(file_path_arg, runtime_arg, *, config=None):
        calls.append(("file", file_path_arg, runtime_arg, config))
        return expected_file

    async def fake_ingest_batch(data_dir_arg, runtime_arg, *, pipeline_config=None):
        calls.append(("batch", data_dir_arg, runtime_arg, pipeline_config))
        return expected_batch

    monkeypatch.setattr(ingest_service, "ingest_file", fake_ingest_file)
    monkeypatch.setattr(ingest_service, "ingest_batch", fake_ingest_batch)

    async def run():
        service = ingest_service.IngestService(runtime)
        file_result = await service.ingest_file(file_path=file_path)
        batch_result = await service.ingest_batch(data_dir=data_dir)
        return file_result, batch_result

    file_result, batch_result = asyncio.run(run())

    assert file_result is expected_file
    assert batch_result is expected_batch
    assert calls == [
        ("file", file_path, runtime, None),
        ("batch", data_dir, runtime, None),
    ]


def test_ingest_file_wraps_missing_collection_as_collection_not_ready(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        dense_embeddings=object(),
        storage_backend=SimpleNamespace(qdrant_store=SimpleNamespace(collection_name="child_chunks_hybrid")),
    )
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    async def fake_run_document_pipeline_async(**kwargs):
        raise UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"{\"status\":{\"error\":\"Not found: Collection `child_chunks_hybrid` doesn't exist!\"}}",
            headers={},
        )

    monkeypatch.setattr(ingest_service, "run_document_pipeline_async", fake_run_document_pipeline_async)

    async def run() -> None:
        await ingest_service.ingest_file(file_path, runtime)

    try:
        asyncio.run(run())
    except CollectionNotReadyError as exc:
        assert exc.collection_name == "child_chunks_hybrid"
    else:
        raise AssertionError("expected CollectionNotReadyError")


def test_ingest_batch_wraps_mongo_errors_as_dependency_unavailable(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        dense_embeddings=object(),
        storage_backend=SimpleNamespace(qdrant_store=SimpleNamespace(collection_name="child_chunks_hybrid")),
    )
    data_dir = tmp_path / "batch"
    data_dir.mkdir()

    async def fake_run_batch_pipeline_async(**kwargs):
        raise ServerSelectionTimeoutError("mongo down")

    monkeypatch.setattr(ingest_service, "run_batch_pipeline_async", fake_run_batch_pipeline_async)

    async def run() -> None:
        await ingest_service.ingest_batch(data_dir, runtime)

    try:
        asyncio.run(run())
    except DependencyUnavailableError as exc:
        assert str(exc) == "MongoDB is unavailable"
    else:
        raise AssertionError("expected DependencyUnavailableError")
