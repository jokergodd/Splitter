import importlib
from pathlib import Path

from langchain_core.documents import Document

from rag_demo import loaders


def _make_fake_loader(document_metadata: dict[str, object], text: str):
    state = {"init_sources": [], "load_calls": 0}

    class _FakeLoader:
        def __init__(self, source: str):
            self.source = source
            state["init_sources"].append(source)

        def load(self) -> list[Document]:
            state["load_calls"] += 1
            return [
                Document(
                    id="doc-1",
                    page_content=text,
                    metadata=document_metadata,
                )
            ]

    return _FakeLoader, state


def test_load_pdf_documents_uses_pymupdf_and_normalizes_metadata(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    fake_loader, state = _make_fake_loader(
        {"page": 3, "source": "legacy", "kind": "pdf"},
        "digital page",
    )
    monkeypatch.setattr(loaders, "PyMuPDFLoader", fake_loader)

    documents = loaders.load_pdf_documents(pdf_path)

    assert state["init_sources"] == [str(pdf_path)]
    assert state["load_calls"] == 1
    assert len(documents) == 1

    document = documents[0]
    assert document.id == "doc-1"
    assert document.page_content == "digital page"
    assert document.metadata["page"] == 3
    assert document.metadata["kind"] == "pdf"
    assert document.metadata["source"] == str(pdf_path)
    assert document.metadata["file_path"] == str(pdf_path)
    assert document.metadata["loader_type"] == "_FakeLoader"
    assert document.metadata["file_type"] == ".pdf"


def test_load_docx_documents_uses_docx_loader_and_normalizes_metadata(monkeypatch, tmp_path):
    docx_path = tmp_path / "sample.docx"
    docx_path.write_bytes(b"docx")

    fake_loader, state = _make_fake_loader(
        {"source": "legacy", "kind": "docx"},
        "word content",
    )
    monkeypatch.setattr(loaders, "Docx2txtLoader", fake_loader)

    documents = loaders.load_docx_documents(docx_path)

    assert state["init_sources"] == [str(docx_path)]
    assert state["load_calls"] == 1
    assert documents[0].page_content == "word content"
    assert documents[0].metadata["kind"] == "docx"
    assert documents[0].metadata["source"] == str(docx_path)
    assert documents[0].metadata["file_path"] == str(docx_path)
    assert documents[0].metadata["loader_type"] == "_FakeLoader"
    assert documents[0].metadata["file_type"] == ".docx"


def test_load_markdown_documents_uses_unstructured_loader_and_normalizes_metadata(monkeypatch, tmp_path):
    markdown_path = tmp_path / "guide.md"
    markdown_path.write_text("# Title", encoding="utf-8")

    fake_loader, state = _make_fake_loader(
        {"category": "markdown"},
        "# Title\n\nBody",
    )
    monkeypatch.setattr(loaders, "UnstructuredMarkdownLoader", fake_loader)

    documents = loaders.load_markdown_documents(markdown_path)

    assert state["init_sources"] == [str(markdown_path)]
    assert state["load_calls"] == 1
    assert documents[0].metadata["category"] == "markdown"
    assert documents[0].metadata["source"] == str(markdown_path)
    assert documents[0].metadata["file_path"] == str(markdown_path)
    assert documents[0].metadata["loader_type"] == "_FakeLoader"
    assert documents[0].metadata["file_type"] == ".md"


def test_load_text_documents_uses_text_loader_and_normalizes_metadata(monkeypatch, tmp_path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("plain text", encoding="utf-8")

    fake_loader, state = _make_fake_loader(
        {"encoding": "utf-8"},
        "plain text",
    )
    monkeypatch.setattr(loaders, "TextLoader", fake_loader)

    documents = loaders.load_text_documents(text_path)

    assert state["init_sources"] == [str(text_path)]
    assert state["load_calls"] == 1
    assert documents[0].metadata["encoding"] == "utf-8"
    assert documents[0].metadata["source"] == str(text_path)
    assert documents[0].metadata["file_path"] == str(text_path)
    assert documents[0].metadata["loader_type"] == "_FakeLoader"
    assert documents[0].metadata["file_type"] == ".txt"


def test_load_documents_dispatches_by_supported_suffix(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    docx_path = tmp_path / "sample.docx"
    markdown_path = tmp_path / "sample.md"
    text_path = tmp_path / "sample.txt"
    for file_path in [pdf_path, docx_path, markdown_path, text_path]:
        file_path.write_text("x", encoding="utf-8")

    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        loaders,
        "load_pdf_documents",
        lambda path: calls.append(("pdf", Path(path))) or [Document(page_content="pdf", metadata={})],
    )
    monkeypatch.setattr(
        loaders,
        "load_docx_documents",
        lambda path: calls.append(("docx", Path(path))) or [Document(page_content="docx", metadata={})],
    )
    monkeypatch.setattr(
        loaders,
        "load_markdown_documents",
        lambda path: calls.append(("md", Path(path))) or [Document(page_content="md", metadata={})],
    )
    monkeypatch.setattr(
        loaders,
        "load_text_documents",
        lambda path: calls.append(("txt", Path(path))) or [Document(page_content="txt", metadata={})],
    )

    assert loaders.load_documents(pdf_path)[0].page_content == "pdf"
    assert loaders.load_documents(docx_path)[0].page_content == "docx"
    assert loaders.load_documents(markdown_path)[0].page_content == "md"
    assert loaders.load_documents(text_path)[0].page_content == "txt"
    assert calls == [
        ("pdf", pdf_path),
        ("docx", docx_path),
        ("md", markdown_path),
        ("txt", text_path),
    ]


def test_load_documents_rejects_unsupported_suffix(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("a,b", encoding="utf-8")

    try:
        loaders.load_documents(file_path)
    except ValueError as exc:
        assert ".csv" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported suffix")


def test_load_pdf_documents_only_depends_on_pymupdf_loader(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    reloaded = importlib.reload(loaders)
    fake_loader, state = _make_fake_loader(
        {"page": 3, "source": "legacy", "kind": "pdf"},
        "digital page",
    )
    monkeypatch.setattr(reloaded, "PyMuPDFLoader", fake_loader)

    documents = reloaded.load_pdf_documents(pdf_path)

    assert state["init_sources"] == [str(pdf_path)]
    assert state["load_calls"] == 1
    assert documents[0].metadata["loader_type"] == "_FakeLoader"
