from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from rag_demo.pipeline import BatchResult, PipelineConfig, PipelineResult
from services import ingest_service


def test_ingest_file_logs_total_duration(caplog, monkeypatch, tmp_path: Path):
    runtime = SimpleNamespace(dense_embeddings=object(), storage_backend=object())
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    expected = PipelineResult(raw_page_count=1, cleaned_page_count=1, parent_chunks=[], child_chunks=[])

    async def fake_run_document_pipeline_async(**kwargs):
        return expected

    monkeypatch.setattr(ingest_service, "run_document_pipeline_async", fake_run_document_pipeline_async)

    async def run() -> PipelineResult:
        return await ingest_service.ingest_file(file_path, runtime, config=PipelineConfig())

    with caplog.at_level("INFO"):
        result = asyncio.run(run())

    assert result is expected
    completed = next(record for record in caplog.records if getattr(record, "event", None) == "ingest.file.completed")
    assert completed.file_path == str(file_path)
    assert completed.duration_ms >= 0


def test_ingest_batch_logs_failures(caplog, monkeypatch, tmp_path: Path):
    runtime = SimpleNamespace(dense_embeddings=object(), storage_backend=object())
    data_dir = tmp_path / "batch"
    data_dir.mkdir()

    async def raise_error(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ingest_service, "run_batch_pipeline_async", raise_error)

    async def run() -> BatchResult:
        return await ingest_service.ingest_batch(data_dir, runtime)

    with caplog.at_level("INFO"):
        try:
            asyncio.run(run())
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

    failed = next(record for record in caplog.records if getattr(record, "event", None) == "ingest.batch.failed")
    assert failed.data_dir == str(data_dir)
    assert failed.error == "boom"
