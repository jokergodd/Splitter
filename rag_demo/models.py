from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document


@dataclass(slots=True)
class ChunkingConfig:
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 200
    child_chunk_size: int | None = None
    child_splitter_mode: str = "auto"
    child_recursive_chunk_size: int = 400
    child_recursive_chunk_overlap: int = 80
    semantic_file_types: tuple[str, ...] = (".pdf", ".docx")
    semantic_min_parent_length: int = 1000
    semantic_max_line_density: float = 0.03
    semantic_chunk_workers: int = 4

    def __post_init__(self) -> None:
        if self.parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be greater than 0")
        if self.parent_chunk_overlap < 0:
            raise ValueError("parent_chunk_overlap must be at least 0")
        if self.parent_chunk_overlap >= self.parent_chunk_size:
            raise ValueError("parent_chunk_overlap must be less than parent_chunk_size")
        if self.semantic_chunk_workers < 1:
            raise ValueError("semantic_chunk_workers must be at least 1")
        if self.child_splitter_mode not in {"recursive", "semantic", "auto"}:
            raise ValueError("child_splitter_mode must be one of: recursive, semantic, auto")
        if self.child_recursive_chunk_size <= 0:
            raise ValueError("child_recursive_chunk_size must be greater than 0")
        if self.child_recursive_chunk_overlap < 0:
            raise ValueError("child_recursive_chunk_overlap must be at least 0")
        if self.child_recursive_chunk_overlap >= self.child_recursive_chunk_size:
            raise ValueError("child_recursive_chunk_overlap must be less than child_recursive_chunk_size")
        if self.semantic_min_parent_length <= 0:
            raise ValueError("semantic_min_parent_length must be greater than 0")
        if not 0 <= self.semantic_max_line_density <= 1:
            raise ValueError("semantic_max_line_density must be between 0 and 1")


@dataclass(slots=True)
class ChunkingResult:
    parent_chunks: list[Document] = field(default_factory=list)
    child_chunks: list[Document] = field(default_factory=list)


@dataclass(slots=True)
class FileProcessingResult:
    file_path: Path
    raw_page_count: int = 0
    cleaned_page_count: int = 0
    parent_chunk_count: int = 0
    child_chunk_count: int = 0
    content_hash: str | None = None
    status: str = "ok"
    skip_reason: str | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchResult:
    directory: Path
    total_files: int
    successful_files: int
    skipped_files: int
    failed_files: int
    files: list[FileProcessingResult] = field(default_factory=list)
