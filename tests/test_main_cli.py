from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import main
from rag_demo.models import BatchResult, FileProcessingResult
from rag_demo.pipeline import PipelineResult


def test_main_file_uses_ingest_service_and_preserves_summary_output(monkeypatch, tmp_path, capsys):
    runtime = object()
    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    parent_doc = Document(page_content="parent", metadata={"parent_id": "parent-0"})
    child_doc = Document(page_content="child", metadata={"child_id": "child-0"})
    expected = PipelineResult(
        raw_page_count=3,
        cleaned_page_count=2,
        parent_chunks=[parent_doc],
        child_chunks=[child_doc],
        status="ok",
        content_hash="hash-123",
    )
    captured: dict[str, object] = {}

    class FakeIngestService:
        def __init__(self, runtime_arg):
            captured["runtime"] = runtime_arg

        async def ingest_file(self, *, file_path, config=None):
            captured["file_path"] = file_path
            captured["config"] = config
            return expected

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: runtime)
    monkeypatch.setattr(main, "IngestService", FakeIngestService)
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--file", str(file_path)])

    assert main.main() == 0

    output = capsys.readouterr().out
    assert "Status: ok" in output
    assert "Content hash: hash-123" in output
    assert "Raw pages: 3" in output
    assert "Cleaned pages: 2" in output
    assert "Parent chunks: 1" in output
    assert "Child chunks: 1" in output
    assert "Sample parent metadata: {'parent_id': 'parent-0'}" in output
    assert "Sample child metadata: {'child_id': 'child-0'}" in output
    assert captured["runtime"] is runtime
    assert captured["file_path"] == file_path
    assert isinstance(captured["config"], main.PipelineConfig)
    assert captured["config"].chunking == main.ChunkingConfig()


def test_main_batch_uses_ingest_service_and_preserves_batch_output(monkeypatch, tmp_path, capsys):
    runtime = object()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    expected = BatchResult(
        directory=data_dir,
        total_files=3,
        successful_files=1,
        skipped_files=1,
        failed_files=1,
        files=[
            FileProcessingResult(
                file_path=data_dir / "ok.pdf",
                raw_page_count=5,
                cleaned_page_count=4,
                parent_chunk_count=2,
                child_chunk_count=6,
                status="ok",
            ),
            FileProcessingResult(
                file_path=data_dir / "skip.txt",
                status="skipped",
                skip_reason="content hash already exists",
            ),
            FileProcessingResult(
                file_path=data_dir / "bad.docx",
                status="failed",
                error="boom",
            ),
        ],
    )
    captured: dict[str, object] = {}

    class FakeIngestService:
        def __init__(self, runtime_arg):
            captured["runtime"] = runtime_arg

        async def ingest_batch(self, *, data_dir, pipeline_config=None):
            captured["data_dir"] = data_dir
            captured["pipeline_config"] = pipeline_config
            return expected

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: runtime)
    monkeypatch.setattr(main, "IngestService", FakeIngestService)
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--data-dir", str(data_dir)])

    assert main.main() == 0

    output = capsys.readouterr().out
    assert "Discovered files: 3" in output
    assert "Successful: 1" in output
    assert "Skipped: 1" in output
    assert "Failed: 1" in output
    assert "ok.pdf raw=5 cleaned=4 parent=2 child=6" in output
    assert "skip.txt skipped reason=content hash already exists" in output
    assert "bad.docx error=boom" in output
    assert captured["runtime"] is runtime
    assert captured["data_dir"] == data_dir
    assert captured["pipeline_config"] is None


def test_main_requires_exactly_one_input_and_does_not_build_runtime(monkeypatch):
    called = {"runtime": False}

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: called.__setitem__("runtime", True))
    monkeypatch.setattr(main.sys, "argv", ["main.py"])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 2
    assert called["runtime"] is False


def test_main_raises_file_not_found_before_service_call(monkeypatch, tmp_path):
    called = {"runtime": False}
    missing = tmp_path / "missing.pdf"

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: called.__setitem__("runtime", True))
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--file", str(missing)])

    with pytest.raises(FileNotFoundError):
        main.main()

    assert called["runtime"] is False
