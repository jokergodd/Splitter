from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .chunking import build_parent_child_chunks
from .cleaning import clean_documents
from .embeddings import CachedEmbeddings
from .loaders import SUPPORTED_FILE_TYPES, load_documents
from .storage import compute_content_hash
from .models import BatchResult, ChunkingConfig, ChunkingResult, FileProcessingResult


@dataclass(slots=True)
class PipelineConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass(slots=True)
class PipelineResult:
    raw_page_count: int
    cleaned_page_count: int
    parent_chunks: list[Document]
    child_chunks: list[Document]
    status: str = "ok"
    content_hash: str | None = None
    skip_reason: str | None = None


_RECOVERABLE_BATCH_ERRORS = (
    FileNotFoundError,
    NotADirectoryError,
    RuntimeError,
    ImportError,
    OSError,
)


def _ensure_cached_embeddings(embeddings):
    if isinstance(embeddings, CachedEmbeddings):
        return embeddings
    return CachedEmbeddings(embeddings)


def discover_supported_files(directory: str | Path) -> list[Path]:
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    supported_files = sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_TYPES
    )
    return supported_files


def run_document_pipeline(
    file_path: str | Path,
    config: PipelineConfig,
    embeddings,
    storage_backend=None,
) -> PipelineResult:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    embeddings = _ensure_cached_embeddings(embeddings)

    content_hash = compute_content_hash(path)
    if storage_backend and storage_backend.mongo_repository.should_skip_hash(content_hash):
        return PipelineResult(
            raw_page_count=0,
            cleaned_page_count=0,
            parent_chunks=[],
            child_chunks=[],
            status="skipped",
            content_hash=content_hash,
            skip_reason="content hash already exists",
        )

    if storage_backend:
        storage_backend.mongo_repository.mark_processing(
            content_hash=content_hash,
            file_path=path,
            file_type=path.suffix.lower(),
            file_size=path.stat().st_size,
        )

    try:
        raw_documents = load_documents(path)
        if not raw_documents:
            raise RuntimeError(f"No documents were loaded from {path}.")

        cleaned_documents = clean_documents(raw_documents)
        if not cleaned_documents:
            raise RuntimeError(f"Cleaning removed all documents from {path}.")

        chunk_result: ChunkingResult = build_parent_child_chunks(
            cleaned_documents,
            config.chunking,
            embeddings,
        )

        if storage_backend:
            parent_ids = storage_backend.mongo_repository.store_parent_chunks(
                content_hash=content_hash,
                file_type=path.suffix.lower(),
                parent_chunks=chunk_result.parent_chunks,
            )
            storage_backend.qdrant_store.store_child_chunks(
                content_hash=content_hash,
                child_chunks=chunk_result.child_chunks,
                embeddings=embeddings,
                sparse_embeddings=storage_backend.sparse_embeddings,
            )
            storage_backend.mongo_repository.mark_completed(
                content_hash=content_hash,
                raw_page_count=len(raw_documents),
                cleaned_page_count=len(cleaned_documents),
                parent_chunk_count=len(chunk_result.parent_chunks),
                child_chunk_count=len(chunk_result.child_chunks),
                parent_ids=parent_ids,
            )
    except Exception as exc:
        if storage_backend:
            storage_backend.mongo_repository.mark_failed(
                content_hash=content_hash,
                error=str(exc),
            )
        raise

    return PipelineResult(
        raw_page_count=len(raw_documents),
        cleaned_page_count=len(cleaned_documents),
        parent_chunks=chunk_result.parent_chunks,
        child_chunks=chunk_result.child_chunks,
        status="ok",
        content_hash=content_hash,
    )


def discover_pdf_files(directory: str | Path) -> list[Path]:
    return [
        file_path
        for file_path in discover_supported_files(directory)
        if file_path.suffix.lower() == ".pdf"
    ]


def run_pdf_pipeline(
    file_path: str | Path,
    config: PipelineConfig,
    embeddings,
    storage_backend=None,
) -> PipelineResult:
    return run_document_pipeline(
        file_path,
        config,
        embeddings,
        storage_backend=storage_backend,
    )


def run_batch_pipeline(
    directory: str | Path,
    embeddings,
    pipeline_config: PipelineConfig | None = None,
    storage_backend=None,
) -> BatchResult:
    path = Path(directory)
    supported_files = discover_supported_files(path)
    pipeline_config = pipeline_config or PipelineConfig()
    embeddings = _ensure_cached_embeddings(embeddings)
    file_results: list[FileProcessingResult] = []

    for file_path in supported_files:
        try:
            pipeline_result = run_document_pipeline(
                file_path,
                PipelineConfig(chunking=pipeline_config.chunking),
                embeddings,
                storage_backend=storage_backend,
            )
        except _RECOVERABLE_BATCH_ERRORS as exc:
            file_results.append(
                FileProcessingResult(
                    file_path=file_path,
                    status="failed",
                    error=str(exc),
                )
            )
            continue

        file_results.append(
            FileProcessingResult(
                file_path=file_path,
                raw_page_count=pipeline_result.raw_page_count,
                cleaned_page_count=pipeline_result.cleaned_page_count,
                parent_chunk_count=len(pipeline_result.parent_chunks),
                child_chunk_count=len(pipeline_result.child_chunks),
                content_hash=pipeline_result.content_hash,
                status=pipeline_result.status,
                skip_reason=pipeline_result.skip_reason,
            )
        )

    successful_files = sum(1 for file_result in file_results if file_result.status == "ok")
    skipped_files = sum(1 for file_result in file_results if file_result.status == "skipped")
    failed_files = sum(1 for file_result in file_results if file_result.status == "failed")
    return BatchResult(
        directory=path,
        total_files=len(file_results),
        successful_files=successful_files,
        skipped_files=skipped_files,
        failed_files=failed_files,
        files=file_results,
    )
