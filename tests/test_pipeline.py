from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import main
from rag_demo.embeddings import CachedEmbeddings
from rag_demo.chunking import ChunkingConfig
from rag_demo.pipeline import PipelineConfig, run_batch_pipeline, run_document_pipeline, run_pdf_pipeline


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class _FakeMongoRepository:
    def __init__(self, should_skip: bool = False):
        self.should_skip = should_skip
        self.processing_calls: list[dict] = []
        self.completed_calls: list[dict] = []
        self.failed_calls: list[dict] = []
        self.parent_store_calls: list[dict] = []

    def should_skip_hash(self, content_hash: str) -> bool:
        return self.should_skip

    def mark_processing(self, **kwargs) -> None:
        self.processing_calls.append(kwargs)

    def mark_completed(self, **kwargs) -> None:
        self.completed_calls.append(kwargs)

    def mark_failed(self, **kwargs) -> None:
        self.failed_calls.append(kwargs)

    def store_parent_chunks(self, **kwargs) -> list[str]:
        self.parent_store_calls.append(kwargs)
        return [parent.metadata["parent_id"] for parent in kwargs["parent_chunks"]]


class _FakeQdrantStore:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    def store_child_chunks(self, **kwargs) -> int:
        if self.error is not None:
            raise self.error
        self.calls.append(kwargs)
        return len(kwargs["child_chunks"])


class _FakeStorageBackend:
    def __init__(self, should_skip: bool = False, child_error: Exception | None = None):
        self.mongo_repository = _FakeMongoRepository(should_skip=should_skip)
        self.qdrant_store = _FakeQdrantStore(error=child_error)
        self.sparse_embeddings = object()


def test_main_prints_pipeline_summary_for_generic_file(monkeypatch, capsys, tmp_path):
    file_path = tmp_path / "demo.docx"
    file_path.write_bytes(b"docx")

    captured = {"runtime": object()}

    async def fake_ingest_file(*, file_path, config):
        captured["file_path"] = file_path
        captured["config"] = config
        return SimpleNamespace(
            raw_page_count=3,
            cleaned_page_count=2,
            parent_chunks=[
                Document(page_content="parent", metadata={"parent_id": "p-1", "topic": "intro"})
            ],
            child_chunks=[Document(page_content="child", metadata={"child_id": "c-1", "parent_id": "p-1"})],
            status="ok",
            content_hash="hash-1",
            skip_reason=None,
        )

    class FakeIngestService:
        def __init__(self, runtime):
            captured["service_runtime"] = runtime

        async def ingest_file(self, *, file_path, config):
            return await fake_ingest_file(file_path=file_path, config=config)

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: captured["runtime"])
    monkeypatch.setattr(main, "IngestService", FakeIngestService)
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--file", str(file_path)])

    assert main.main() == 0

    output = capsys.readouterr().out
    assert "Raw pages: 3" in output
    assert "Cleaned pages: 2" in output
    assert "Parent chunks: 1" in output
    assert "Child chunks: 1" in output
    assert "Sample parent metadata:" in output
    assert "parent_id" in output
    assert "topic" in output
    assert "Sample child metadata:" in output
    assert "child_id" in output
    assert "parent_id" in output
    assert captured["file_path"] == file_path
    assert captured["config"] == PipelineConfig(chunking=ChunkingConfig())
    assert captured["service_runtime"] is captured["runtime"]


def test_main_fails_fast_on_missing_file_before_embeddings(monkeypatch, tmp_path):
    file_path = tmp_path / "missing.docx"

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: (_ for _ in ()).throw(AssertionError("runtime should not initialize")))
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--file", str(file_path)])

    with pytest.raises(FileNotFoundError) as excinfo:
        main.main()

    assert str(file_path) in str(excinfo.value)


def test_main_runs_batch_mode_for_data_directory(monkeypatch, capsys, tmp_path):
    data_dir = tmp_path / "batch"
    data_dir.mkdir()
    first_file = data_dir / "a.pdf"
    second_file = data_dir / "b.md"
    first_file.write_bytes(b"%PDF-1.4\n")
    second_file.write_text("# hello", encoding="utf-8")

    captured = {"runtime": object()}

    async def fake_ingest_batch(*, data_dir):
        captured["directory"] = data_dir
        return SimpleNamespace(
            total_files=2,
            successful_files=1,
            skipped_files=0,
            failed_files=1,
            files=[
                SimpleNamespace(
                    file_path=first_file,
                    status="ok",
                    raw_page_count=3,
                    cleaned_page_count=2,
                    parent_chunk_count=1,
                    child_chunk_count=2,
                    content_hash="hash-1",
                    skip_reason=None,
                    error=None,
                ),
                SimpleNamespace(
                    file_path=second_file,
                    status="failed",
                    raw_page_count=0,
                    cleaned_page_count=0,
                    parent_chunk_count=0,
                    child_chunk_count=0,
                    content_hash="hash-2",
                    skip_reason=None,
                    error="parse failed",
                ),
            ],
        )

    class FakeIngestService:
        def __init__(self, runtime):
            captured["service_runtime"] = runtime

        async def ingest_batch(self, *, data_dir):
            return await fake_ingest_batch(data_dir=data_dir)

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: captured["runtime"])
    monkeypatch.setattr(main, "IngestService", FakeIngestService)
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--data-dir", str(data_dir)])

    assert main.main() == 0

    output = capsys.readouterr().out
    assert "Discovered files: 2" in output
    assert "Successful: 1" in output
    assert "Failed: 1" in output
    assert f"{first_file.name} raw=3 cleaned=2 parent=1 child=2" in output
    assert f"{second_file.name} error=parse failed" in output
    assert captured["directory"] == data_dir
    assert captured["service_runtime"] is captured["runtime"]


def test_main_prints_empty_batch_summary_for_directory_without_supported_files(monkeypatch, capsys, tmp_path):
    data_dir = tmp_path / "batch"
    data_dir.mkdir()

    captured = {"runtime": object()}

    async def fake_ingest_batch(*, data_dir):
        captured["directory"] = data_dir
        return SimpleNamespace(
            total_files=0,
            successful_files=0,
            skipped_files=0,
            failed_files=0,
            files=[],
        )

    class FakeIngestService:
        def __init__(self, runtime):
            captured["service_runtime"] = runtime

        async def ingest_batch(self, *, data_dir):
            return await fake_ingest_batch(data_dir=data_dir)

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: captured["runtime"])
    monkeypatch.setattr(main, "IngestService", FakeIngestService)
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--data-dir", str(data_dir)])

    assert main.main() == 0

    output = capsys.readouterr().out
    assert "Discovered files: 0" in output
    assert "Successful: 0" in output
    assert "Failed: 0" in output
    assert captured["directory"] == data_dir
    assert captured["service_runtime"] is captured["runtime"]


def test_main_prints_skipped_summary(monkeypatch, capsys, tmp_path):
    data_dir = tmp_path / "batch"
    data_dir.mkdir()
    skipped_file = data_dir / "dup.txt"
    skipped_file.write_text("same", encoding="utf-8")

    async def fake_ingest_batch(*, data_dir):
        return SimpleNamespace(
            total_files=1,
            successful_files=0,
            skipped_files=1,
            failed_files=0,
            files=[
                SimpleNamespace(
                    file_path=skipped_file,
                    status="skipped",
                    raw_page_count=0,
                    cleaned_page_count=0,
                    parent_chunk_count=0,
                    child_chunk_count=0,
                    content_hash="hash-1",
                    skip_reason="content hash already exists",
                    error=None,
                )
            ],
        )

    class FakeIngestService:
        def __init__(self, runtime):
            self.runtime = runtime

        async def ingest_batch(self, *, data_dir):
            return await fake_ingest_batch(data_dir=data_dir)

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: object())
    monkeypatch.setattr(main, "IngestService", FakeIngestService)
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--data-dir", str(data_dir)])

    assert main.main() == 0

    output = capsys.readouterr().out
    assert "Skipped: 1" in output
    assert f"{skipped_file.name} skipped reason=content hash already exists" in output


def test_main_fails_fast_on_missing_data_directory_before_embeddings(monkeypatch, tmp_path):
    data_dir = tmp_path / "missing-batch"

    monkeypatch.setattr(main, "get_ingest_runtime", lambda: (_ for _ in ()).throw(AssertionError("runtime should not initialize")))
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--data-dir", str(data_dir)])

    with pytest.raises(FileNotFoundError) as excinfo:
        main.main()

    assert str(data_dir) in str(excinfo.value)


def test_main_rejects_file_and_data_dir_together(monkeypatch, capsys, tmp_path):
    file_path = tmp_path / "demo.pdf"
    data_dir = tmp_path / "batch"
    file_path.write_bytes(b"%PDF-1.4\n")
    data_dir.mkdir()

    monkeypatch.setattr(
        main.sys,
        "argv",
        ["main.py", "--file", str(file_path), "--data-dir", str(data_dir)],
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2
    assert "choose exactly one of --file or --data-dir" in capsys.readouterr().err


def test_main_rejects_missing_input(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "argv", ["main.py"])

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2
    assert "choose exactly one of --file or --data-dir" in capsys.readouterr().err


def test_run_document_pipeline_returns_counts_and_traceable_children(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content", encoding="utf-8")

    raw_documents = [
        Document(page_content="   \n\n", metadata={"page": 1}),
        Document(page_content="重复中文重复中文重复中文" * 20, metadata={"page": 2}),
    ]

    monkeypatch.setattr(
        "rag_demo.pipeline.load_documents",
        lambda file_path: raw_documents,
    )
    monkeypatch.setattr("rag_demo.pipeline.compute_content_hash", lambda file_path: "hash-1")
    storage_backend = _FakeStorageBackend()

    result = run_document_pipeline(
        file_path,
        PipelineConfig(
            chunking=ChunkingConfig(parent_chunk_size=220, parent_chunk_overlap=30),
        ),
        _FakeEmbeddings(),
        storage_backend=storage_backend,
    )

    assert result.status == "ok"
    assert result.content_hash == "hash-1"
    assert result.raw_page_count == 2
    assert result.cleaned_page_count == 1
    assert result.parent_chunks
    assert result.child_chunks
    assert result.child_chunks[0].metadata["parent_id"] == result.parent_chunks[0].metadata["parent_id"]
    assert storage_backend.mongo_repository.processing_calls
    assert storage_backend.mongo_repository.completed_calls
    assert storage_backend.qdrant_store.calls
    assert storage_backend.qdrant_store.calls[0]["sparse_embeddings"] is storage_backend.sparse_embeddings


def test_run_document_pipeline_skips_completed_duplicate(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("duplicate", encoding="utf-8")

    monkeypatch.setattr("rag_demo.pipeline.compute_content_hash", lambda file_path: "hash-dup")
    storage_backend = _FakeStorageBackend(should_skip=True)

    result = run_document_pipeline(
        file_path,
        PipelineConfig(),
        _FakeEmbeddings(),
        storage_backend=storage_backend,
    )

    assert result.status == "skipped"
    assert result.content_hash == "hash-dup"
    assert result.skip_reason == "content hash already exists"
    assert result.raw_page_count == 0
    assert result.cleaned_page_count == 0
    assert storage_backend.mongo_repository.processing_calls == []
    assert storage_backend.qdrant_store.calls == []


def test_run_document_pipeline_retries_failed_duplicate(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("retry", encoding="utf-8")

    monkeypatch.setattr("rag_demo.pipeline.compute_content_hash", lambda file_path: "hash-retry")
    monkeypatch.setattr(
        "rag_demo.pipeline.load_documents",
        lambda file_path: [Document(page_content="retry body " * 40, metadata={"file_type": ".txt"})],
    )
    storage_backend = _FakeStorageBackend(should_skip=False)

    result = run_document_pipeline(
        file_path,
        PipelineConfig(),
        _FakeEmbeddings(),
        storage_backend=storage_backend,
    )

    assert result.status == "ok"
    assert storage_backend.mongo_repository.processing_calls
    assert storage_backend.mongo_repository.completed_calls


def test_run_document_pipeline_marks_failed_when_child_storage_raises(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("boom", encoding="utf-8")

    monkeypatch.setattr("rag_demo.pipeline.compute_content_hash", lambda file_path: "hash-fail")
    monkeypatch.setattr(
        "rag_demo.pipeline.load_documents",
        lambda file_path: [Document(page_content="retry body " * 40, metadata={"file_type": ".txt"})],
    )
    storage_backend = _FakeStorageBackend(child_error=RuntimeError("qdrant write failed"))

    with pytest.raises(RuntimeError, match="qdrant write failed"):
        run_document_pipeline(
            file_path,
            PipelineConfig(),
            _FakeEmbeddings(),
            storage_backend=storage_backend,
        )

    assert storage_backend.mongo_repository.failed_calls == [
        {"content_hash": "hash-fail", "error": "qdrant write failed"}
    ]


def test_run_document_pipeline_raises_when_no_documents_loaded(monkeypatch, tmp_path):
    file_path = tmp_path / "empty.md"
    file_path.write_text("# empty", encoding="utf-8")

    monkeypatch.setattr("rag_demo.pipeline.load_documents", lambda file_path: [])
    monkeypatch.setattr("rag_demo.pipeline.compute_content_hash", lambda file_path: "hash-empty")

    with pytest.raises(RuntimeError, match="No documents were loaded"):
        run_document_pipeline(
            file_path,
            PipelineConfig(),
            _FakeEmbeddings(),
            storage_backend=_FakeStorageBackend(),
        )


def test_run_document_pipeline_raises_when_cleaning_removes_all_documents(monkeypatch, tmp_path):
    file_path = tmp_path / "blank.txt"
    file_path.write_text("   \n\n", encoding="utf-8")

    monkeypatch.setattr(
        "rag_demo.pipeline.load_documents",
        lambda file_path: [Document(page_content="   \n\n", metadata={"page": 1})],
    )
    monkeypatch.setattr("rag_demo.pipeline.compute_content_hash", lambda file_path: "hash-blank")

    with pytest.raises(RuntimeError, match="Cleaning removed all documents"):
        run_document_pipeline(
            file_path,
            PipelineConfig(),
            _FakeEmbeddings(),
            storage_backend=_FakeStorageBackend(),
        )


def test_run_document_pipeline_writes_hybrid_child_chunks_when_storage_backend_is_present(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content", encoding="utf-8")

    monkeypatch.setattr("rag_demo.pipeline.compute_content_hash", lambda file_path: "hash-hybrid")
    monkeypatch.setattr(
        "rag_demo.pipeline.load_documents",
        lambda file_path: [Document(page_content="retry body " * 40, metadata={"file_type": ".txt"})],
    )
    storage_backend = _FakeStorageBackend()

    result = run_document_pipeline(
        file_path,
        PipelineConfig(),
        _FakeEmbeddings(),
        storage_backend=storage_backend,
    )

    assert result.status == "ok"
    assert storage_backend.qdrant_store.calls
    assert storage_backend.qdrant_store.calls[0]["sparse_embeddings"] is storage_backend.sparse_embeddings


def test_run_pdf_pipeline_forwards_storage_backend(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    storage_backend = object()
    captured: dict[str, object] = {}

    def fake_run_document_pipeline(file_path, config, embeddings, storage_backend=None):
        captured["file_path"] = file_path
        captured["config"] = config
        captured["embeddings"] = embeddings
        captured["storage_backend"] = storage_backend
        return SimpleNamespace(
            raw_page_count=1,
            cleaned_page_count=1,
            parent_chunks=[],
            child_chunks=[],
            status="ok",
            content_hash="hash-pdf",
            skip_reason=None,
        )

    monkeypatch.setattr("rag_demo.pipeline.run_document_pipeline", fake_run_document_pipeline)

    result = run_pdf_pipeline(
        file_path,
        PipelineConfig(),
        _FakeEmbeddings(),
        storage_backend=storage_backend,
    )

    assert result.status == "ok"
    assert captured["file_path"] == file_path
    assert captured["storage_backend"] is storage_backend


def test_run_batch_pipeline_reuses_same_embeddings_object_for_each_file(monkeypatch, tmp_path):
    data_dir = tmp_path / "batch"
    data_dir.mkdir()
    first_file = data_dir / "a.pdf"
    second_file = data_dir / "b.md"
    first_file.write_bytes(b"%PDF-1.4\n")
    second_file.write_text("# hello", encoding="utf-8")

    embeddings = _FakeEmbeddings()
    seen_embeddings: list[object] = []

    monkeypatch.setattr("rag_demo.pipeline.discover_supported_files", lambda directory: [first_file, second_file])

    def fake_run_document_pipeline(file_path, config, embeddings, storage_backend=None):
        seen_embeddings.append(embeddings)
        return SimpleNamespace(
            raw_page_count=1,
            cleaned_page_count=1,
            parent_chunks=[],
            child_chunks=[],
            status="ok",
            content_hash="hash-1",
            skip_reason=None,
        )

    monkeypatch.setattr("rag_demo.pipeline.run_document_pipeline", fake_run_document_pipeline)

    result = run_batch_pipeline(data_dir, embeddings, storage_backend=_FakeStorageBackend())

    assert result.total_files == 2
    assert len({id(item) for item in seen_embeddings}) == 1
    assert isinstance(seen_embeddings[0], CachedEmbeddings)
