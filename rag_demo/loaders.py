from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

SUPPORTED_FILE_TYPES = {".pdf", ".docx", ".md", ".txt"}


def _normalize_document_metadata(document: Document, file_path: Path, loader_type: str) -> Document:
    metadata = dict(document.metadata)
    metadata["source"] = str(file_path)
    metadata["file_path"] = str(file_path)
    metadata["loader_type"] = loader_type
    metadata["file_type"] = file_path.suffix.lower()
    return Document(
        page_content=document.page_content,
        metadata=metadata,
        id=document.id,
    )


def _load_with_loader(file_path: str | Path, loader_class) -> list[Document]:
    path = Path(file_path)
    loader = loader_class(str(path))
    documents = loader.load()
    loader_type = type(loader).__name__
    return [
        _normalize_document_metadata(document, path, loader_type)
        for document in documents
    ]


def load_pdf_documents(file_path: str | Path) -> list[Document]:
    return _load_with_loader(file_path, PyMuPDFLoader)


def load_docx_documents(file_path: str | Path) -> list[Document]:
    return _load_with_loader(file_path, Docx2txtLoader)


def load_markdown_documents(file_path: str | Path) -> list[Document]:
    return _load_with_loader(file_path, UnstructuredMarkdownLoader)


def load_text_documents(file_path: str | Path) -> list[Document]:
    return _load_with_loader(file_path, TextLoader)


def load_documents(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf_documents(path)
    if suffix == ".docx":
        return load_docx_documents(path)
    if suffix == ".md":
        return load_markdown_documents(path)
    if suffix == ".txt":
        return load_text_documents(path)

    raise ValueError(f"Unsupported file type: {suffix}")
