import re

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

import rag_demo.chunking as chunking
from rag_demo.chunking import build_parent_child_chunks
from rag_demo.models import ChunkingConfig


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(sum(ord(char) for char in text) % 97)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(ord(char) for char in text) % 97)]


class _FakeSemanticChunker:
    last_sentence_split_regex = None

    def __init__(self, embeddings, sentence_split_regex: str, min_chunk_size=None):
        self.embeddings = embeddings
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        type(self).last_sentence_split_regex = sentence_split_regex

    def split_documents(self, documents: list[Document]) -> list[Document]:
        child_documents: list[Document] = []

        for document in documents:
            sentences = [sentence for sentence in re.split(self.sentence_split_regex, document.page_content) if sentence]
            for sentence in sentences:
                child_documents.append(
                    Document(page_content=sentence, metadata=dict(document.metadata))
                )

        return child_documents


class _ConflictingMetadataSemanticChunker:
    def __init__(self, embeddings, sentence_split_regex: str, min_chunk_size=None):
        self.embeddings = embeddings
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size

    def split_documents(self, documents: list[Document]) -> list[Document]:
        parent = documents[0]
        return [
            Document(
                page_content="child body",
                metadata={
                    "parent_id": "child-parent-id",
                    "parent_index": 999,
                    "child_id": "child-child-id",
                    "child_index": 888,
                    "parent_text": "child parent text",
                    "source": "child-source",
                },
            )
            if parent.page_content
            else Document(page_content="child body", metadata={})
        ]


class _FakeRecursiveCharacterTextSplitter:
    init_args = []
    split_calls = []

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        type(self).init_args.append((chunk_size, chunk_overlap))

    def split_documents(self, documents: list[Document]) -> list[Document]:
        type(self).split_calls.append([document.page_content for document in documents])
        chunks: list[Document] = []
        for document in documents:
            text = document.page_content
            midpoint = max(1, len(text) // 2)
            first_text = text[:midpoint]
            second_text = text[midpoint:]
            chunks.append(Document(page_content=first_text, metadata=dict(document.metadata)))
            if second_text:
                chunks.append(Document(page_content=second_text, metadata=dict(document.metadata)))
        return chunks


class _TrackingSemanticChildChunker:
    init_args = []
    split_calls = []

    def __init__(self, embeddings, sentence_split_regex: str, min_chunk_size=None):
        self.embeddings = embeddings
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        type(self).init_args.append((sentence_split_regex, min_chunk_size))

    def split_documents(self, documents: list[Document]) -> list[Document]:
        type(self).split_calls.append([document.page_content for document in documents])
        return [
            Document(
                page_content=f"semantic-child:{document.page_content}",
                metadata=dict(document.metadata),
            )
            for document in documents
        ]


class _TrackingRecursiveChildSplitter:
    init_args = []
    split_calls = []

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        type(self).init_args.append((chunk_size, chunk_overlap))

    def split_documents(self, documents: list[Document]) -> list[Document]:
        type(self).split_calls.append([document.page_content for document in documents])
        return [
            Document(
                page_content=f"recursive-child:{document.page_content}",
                metadata=dict(document.metadata),
            )
            for document in documents
        ]


class _FakeMarkdownHeaderTextSplitter:
    split_text_calls = []
    split_text_responses = None

    def __init__(self, headers_to_split_on, strip_headers=False):
        self.headers_to_split_on = headers_to_split_on
        self.strip_headers = strip_headers

    def _with_headers_preserved(self, document: Document) -> Document:
        if self.strip_headers:
            return document

        header_lines: list[str] = []
        for key in sorted(document.metadata):
            if not key.startswith("header_"):
                continue

            level = int(key.split("_", 1)[1])
            header_lines.append(f"{'#' * level} {document.metadata[key]}")

        if not header_lines:
            return document

        header_prefix = "\n".join(header_lines)
        if document.page_content.startswith(header_prefix):
            return document

        return Document(
            page_content=f"{header_prefix}\n{document.page_content}",
            metadata=dict(document.metadata),
        )

    def split_text(self, text: str) -> list[Document]:
        type(self).split_text_calls.append(text)
        if type(self).split_text_responses:
            return [
                self._with_headers_preserved(document)
                for document in type(self).split_text_responses.pop(0)
            ]
        if "# " in text:
            return [
                self._with_headers_preserved(
                    Document(
                        page_content="Intro body",
                        metadata={"header_1": "Intro"},
                    )
                ),
                self._with_headers_preserved(
                    Document(
                        page_content="Details body",
                        metadata={"header_1": "Details"},
                    )
                ),
            ]
        return []


class _RecordingExecutor:
    created_max_workers = []
    submitted_calls = []

    def __init__(self, max_workers: int):
        type(self).created_max_workers.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        type(self).submitted_calls.append((args, kwargs))

        class _ImmediateFuture:
            def result(self_inner):
                return fn(*args, **kwargs)

        return _ImmediateFuture()


class _OutOfOrderExecutor:
    created_max_workers = []
    submitted_parent_ids = []
    completed_parent_ids = []

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self._futures = []
        self._drained = False
        type(self).created_max_workers.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        future = _OutOfOrderFuture(self, fn, args, kwargs)
        self._futures.append(future)
        type(self).submitted_parent_ids.append(args[0].page_content)
        return future

    def _drain(self):
        if self._drained:
            return

        for future in reversed(self._futures):
            future.set_result(future.fn(*future.args, **future.kwargs))
            type(self).completed_parent_ids.append(future.args[0].page_content)

        self._drained = True


class _OutOfOrderFuture:
    def __init__(self, executor, fn, args, kwargs):
        self.executor = executor
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self._result = None

    def set_result(self, result):
        self._result = result

    def result(self):
        self.executor._drain()
        return self._result


def _reset_child_routing_trackers() -> None:
    _TrackingSemanticChildChunker.init_args = []
    _TrackingSemanticChildChunker.split_calls = []
    _TrackingRecursiveChildSplitter.init_args = []
    _TrackingRecursiveChildSplitter.split_calls = []


def _reset_markdown_fallback_trackers() -> None:
    _FakeMarkdownHeaderTextSplitter.split_text_calls = []
    _FakeMarkdownHeaderTextSplitter.split_text_responses = None
    _FakeRecursiveCharacterTextSplitter.init_args = []
    _FakeRecursiveCharacterTextSplitter.split_calls = []
    _TrackingSemanticChildChunker.init_args = []
    _TrackingSemanticChildChunker.split_calls = []
    _TrackingRecursiveChildSplitter.init_args = []
    _TrackingRecursiveChildSplitter.split_calls = []
    _RecordingExecutor.created_max_workers = []
    _RecordingExecutor.submitted_calls = []


def _flatten_split_calls(split_calls: list[list[str]]) -> list[str]:
    return [text for call in split_calls for text in call]


def test_build_parent_child_chunks_propagates_parent_metadata():
    text = "\u8fd9\u662f\u7528\u4e8e\u6d4b\u8bd5\u7236\u5b50\u5206\u5757\u7684\u4e2d\u6587\u5185\u5bb9\u3002"
    long_text = text * 40
    documents = [
        Document(
            page_content=long_text,
            metadata={
                "page": 3,
                "source": "demo.pdf",
                "source_file": "demo.pdf",
                "file_path": "demo.pdf",
                "loader_type": "digital",
            },
        )
    ]
    config = ChunkingConfig(parent_chunk_size=80, parent_chunk_overlap=10, child_chunk_size=40)
    embeddings = _FakeEmbeddings()

    result = build_parent_child_chunks(documents, config, embeddings)

    assert result.parent_chunks
    assert result.child_chunks

    first_parent = result.parent_chunks[0]
    first_child = result.child_chunks[0]

    assert "parent_id" in first_parent.metadata
    assert first_child.metadata["parent_id"] == first_parent.metadata["parent_id"]
    assert first_child.metadata["parent_text"] == first_parent.page_content
    assert first_child.metadata["page"] == 3


def test_build_parent_child_chunks_preserves_deterministic_child_metadata(monkeypatch):
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _ConflictingMetadataSemanticChunker)

    documents = [
        Document(
            page_content="\u8fd9\u662f\u7528\u4e8e\u6d4b\u8bd5\u7236\u5b50\u5206\u5757\u7684\u4e2d\u6587\u5185\u5bb9\u3002",
            metadata={
                "page": 3,
                "source": "demo.pdf",
                "file_path": "demo.pdf",
                "loader_type": "digital",
            },
        )
    ]

    result = build_parent_child_chunks(
        documents,
        ChunkingConfig(
            parent_chunk_size=200,
            parent_chunk_overlap=0,
            child_chunk_size=20,
            child_splitter_mode="semantic",
        ),
        _FakeEmbeddings(),
    )

    child = result.child_chunks[0]
    parent = result.parent_chunks[0]

    assert child.metadata["parent_id"] == parent.metadata["parent_id"]
    assert child.metadata["parent_index"] == parent.metadata["parent_index"]
    assert child.metadata["child_id"] == f"{parent.metadata['parent_id']}-child-0"
    assert child.metadata["child_index"] == 0
    assert child.metadata["parent_text"] == parent.page_content


def test_build_parent_child_chunks_uses_configured_chinese_sentence_regex_contract(monkeypatch):
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _FakeSemanticChunker)
    chinese_sentences = ["第一句。", "第二句！", "第三句？", "第四句。", "第五句！"]

    documents = [
        Document(
            page_content="".join(chinese_sentences),
            metadata={"page": 3, "source": "demo.pdf", "file_path": "demo.pdf", "loader_type": "digital"},
        )
    ]

    result = build_parent_child_chunks(
        documents,
        ChunkingConfig(
            parent_chunk_size=200,
            parent_chunk_overlap=0,
            child_chunk_size=20,
            child_splitter_mode="semantic",
        ),
        _FakeEmbeddings(),
    )

    assert _FakeSemanticChunker.last_sentence_split_regex == r"(?<=[\u3002\uff01\uff1f.!?])\s*"
    assert [child.page_content for child in result.child_chunks] == chinese_sentences
    assert result.child_chunks[0].metadata["parent_id"] == result.parent_chunks[0].metadata["parent_id"]


def test_build_parent_child_chunks_raises_clear_error_when_semantic_chunker_is_missing(monkeypatch):
    def _missing_semantic_chunker(*args, **kwargs):
        raise ImportError("langchain-experimental is missing")

    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", _missing_semantic_chunker)

    documents = [
        Document(
            page_content="\u8fd9\u662f\u7528\u4e8e\u6d4b\u8bd5\u7236\u5b50\u5206\u5757\u7684\u4e2d\u6587\u5185\u5bb9\u3002",
            metadata={"page": 3, "source": "demo.pdf", "file_path": "demo.pdf", "loader_type": "digital"},
        )
    ]

    try:
        build_parent_child_chunks(
            documents,
            ChunkingConfig(child_splitter_mode="semantic"),
            _FakeEmbeddings(),
        )
    except RuntimeError as exc:
        assert "SemanticChunker" in str(exc)
        assert "langchain-experimental" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_build_parent_child_chunks_uses_recursive_parent_splitter_for_docx_and_txt(monkeypatch):
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _FakeSemanticChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _FakeRecursiveCharacterTextSplitter)

    documents = [
        Document(
            page_content="word document content " * 10,
            metadata={"file_type": ".docx", "source": "demo.docx", "file_path": "demo.docx"},
        ),
        Document(
            page_content="plain text content " * 10,
            metadata={"file_type": ".txt", "source": "demo.txt", "file_path": "demo.txt"},
        ),
    ]

    result = build_parent_child_chunks(
        documents,
        ChunkingConfig(
            parent_chunk_size=50,
            parent_chunk_overlap=5,
            child_chunk_size=20,
            child_splitter_mode="semantic",
        ),
        _FakeEmbeddings(),
    )

    assert _FakeRecursiveCharacterTextSplitter.init_args[-1] == (50, 5)
    assert len(_FakeRecursiveCharacterTextSplitter.split_calls) == 2
    assert len(result.parent_chunks) >= 2
    assert all("parent_id" in parent.metadata for parent in result.parent_chunks)


def test_build_parent_child_chunks_uses_markdown_headers_for_markdown_parents(monkeypatch):
    _FakeMarkdownHeaderTextSplitter.split_text_calls = []
    _FakeMarkdownHeaderTextSplitter.split_text_responses = None
    _FakeRecursiveCharacterTextSplitter.init_args = []
    _FakeRecursiveCharacterTextSplitter.split_calls = []
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _FakeSemanticChunker)
    monkeypatch.setattr(chunking, "MarkdownHeaderTextSplitter", _FakeMarkdownHeaderTextSplitter)

    documents = [
        Document(
            page_content="# Intro\nBody\n## Details\nMore",
            metadata={"file_type": ".md", "source": "guide.md", "file_path": "guide.md"},
        )
    ]

    result = build_parent_child_chunks(
        documents,
        ChunkingConfig(parent_chunk_size=100, parent_chunk_overlap=0, child_chunk_size=10),
        _FakeEmbeddings(),
    )

    assert _FakeMarkdownHeaderTextSplitter.split_text_calls == ["# Intro\nBody\n## Details\nMore"]
    assert len(result.parent_chunks) == 2
    assert result.parent_chunks[0].page_content == "# Intro\nIntro body"
    assert result.parent_chunks[1].page_content == "# Details\nDetails body"
    assert result.parent_chunks[0].metadata["header_1"] == "Intro"
    assert result.parent_chunks[1].metadata["header_1"] == "Details"


def test_build_parent_child_chunks_keeps_markdown_without_headers(monkeypatch):
    _FakeMarkdownHeaderTextSplitter.split_text_calls = []
    _FakeMarkdownHeaderTextSplitter.split_text_responses = None
    _FakeRecursiveCharacterTextSplitter.init_args = []
    _FakeRecursiveCharacterTextSplitter.split_calls = []
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _FakeSemanticChunker)
    monkeypatch.setattr(chunking, "MarkdownHeaderTextSplitter", _FakeMarkdownHeaderTextSplitter)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _FakeRecursiveCharacterTextSplitter)

    documents = [
        Document(
            page_content="markdown without headings but still useful content",
            metadata={"file_type": ".md", "source": "notes.md", "file_path": "notes.md"},
        )
    ]

    result = build_parent_child_chunks(
        documents,
        ChunkingConfig(parent_chunk_size=60, parent_chunk_overlap=0, child_chunk_size=10),
        _FakeEmbeddings(),
    )

    assert _FakeMarkdownHeaderTextSplitter.split_text_calls == ["markdown without headings but still useful content"]
    assert result.parent_chunks
    assert result.parent_chunks[0].page_content


def test_build_parent_child_chunks_keeps_short_markdown_section_without_fallback(monkeypatch):
    _reset_markdown_fallback_trackers()
    _FakeMarkdownHeaderTextSplitter.split_text_responses = [
        [
            Document(
                page_content="Short note",
                metadata={"header_1": "Intro"},
            )
        ]
    ]
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "MarkdownHeaderTextSplitter", _FakeMarkdownHeaderTextSplitter)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(chunking, "ThreadPoolExecutor", _RecordingExecutor, raising=False)

    result = build_parent_child_chunks(
        [Document(page_content="# Intro\nShort note", metadata={"file_type": ".md", "source": "guide.md"})],
        ChunkingConfig(
            child_splitter_mode="semantic",
            parent_chunk_size=100,
            parent_chunk_overlap=0,
            child_chunk_size=5,
        ),
        _FakeEmbeddings(),
    )

    assert _FakeMarkdownHeaderTextSplitter.split_text_calls == ["# Intro\nShort note"]
    assert _TrackingRecursiveChildSplitter.init_args == []
    assert _TrackingRecursiveChildSplitter.split_calls == []
    assert len(result.parent_chunks) == 1
    assert result.parent_chunks[0].page_content == "# Intro\nShort note"
    assert result.parent_chunks[0].metadata["header_1"] == "Intro"


def test_build_parent_child_chunks_triggers_recursive_fallback_for_long_markdown_section(monkeypatch):
    _reset_markdown_fallback_trackers()
    long_markdown_body = "This markdown section is long enough to require recursive fallback. " * 5
    _FakeMarkdownHeaderTextSplitter.split_text_responses = [
        [
            Document(
                page_content=long_markdown_body,
                metadata={"header_1": "Intro"},
            )
        ]
    ]
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "MarkdownHeaderTextSplitter", _FakeMarkdownHeaderTextSplitter)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _FakeRecursiveCharacterTextSplitter)
    monkeypatch.setattr(chunking, "ThreadPoolExecutor", _RecordingExecutor, raising=False)

    result = build_parent_child_chunks(
        [Document(page_content="# Intro\n" + long_markdown_body, metadata={"file_type": ".md", "source": "guide.md"})],
        ChunkingConfig(
            child_splitter_mode="semantic",
            parent_chunk_size=100,
            parent_chunk_overlap=0,
            child_chunk_size=5,
        ),
        _FakeEmbeddings(),
    )

    assert _FakeRecursiveCharacterTextSplitter.init_args == [(100, 0)]
    assert _FakeRecursiveCharacterTextSplitter.split_calls == [["# Intro\n" + long_markdown_body]]
    assert len(result.parent_chunks) > 1
    assert all(parent.metadata["header_1"] == "Intro" for parent in result.parent_chunks)


def test_build_parent_child_chunks_triggers_recursive_fallback_for_long_markdown_without_headers(monkeypatch):
    _reset_markdown_fallback_trackers()
    long_markdown_body = "This markdown note has no heading but still needs recursive fallback. " * 5
    _FakeMarkdownHeaderTextSplitter.split_text_responses = [[]]
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "MarkdownHeaderTextSplitter", _FakeMarkdownHeaderTextSplitter)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _FakeRecursiveCharacterTextSplitter)
    monkeypatch.setattr(chunking, "ThreadPoolExecutor", _RecordingExecutor, raising=False)

    result = build_parent_child_chunks(
        [Document(page_content=long_markdown_body, metadata={"file_type": ".md", "source": "notes.md"})],
        ChunkingConfig(
            child_splitter_mode="semantic",
            parent_chunk_size=100,
            parent_chunk_overlap=0,
            child_chunk_size=5,
        ),
        _FakeEmbeddings(),
    )

    assert _FakeMarkdownHeaderTextSplitter.split_text_calls == [long_markdown_body]
    assert _FakeRecursiveCharacterTextSplitter.init_args == [(100, 0)]
    assert _FakeRecursiveCharacterTextSplitter.split_calls == [[long_markdown_body]]
    assert len(result.parent_chunks) > 1
    assert all("header_1" not in parent.metadata for parent in result.parent_chunks)


def test_build_parent_child_chunks_preserves_header_metadata_after_recursive_markdown_fallback(monkeypatch):
    _reset_markdown_fallback_trackers()
    long_markdown_body = "This markdown section is long enough to require recursive fallback. " * 5
    _FakeMarkdownHeaderTextSplitter.split_text_responses = [
        [
            Document(
                page_content=long_markdown_body,
                metadata={"header_1": "Intro", "header_2": "Details"},
            )
        ]
    ]
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "MarkdownHeaderTextSplitter", _FakeMarkdownHeaderTextSplitter)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _FakeRecursiveCharacterTextSplitter)
    monkeypatch.setattr(chunking, "ThreadPoolExecutor", _RecordingExecutor, raising=False)

    result = build_parent_child_chunks(
        [Document(page_content="# Intro\n## Details\n" + long_markdown_body, metadata={"file_type": ".md", "source": "guide.md"})],
        ChunkingConfig(
            child_splitter_mode="semantic",
            parent_chunk_size=100,
            parent_chunk_overlap=0,
            child_chunk_size=5,
        ),
        _FakeEmbeddings(),
    )

    assert _FakeMarkdownHeaderTextSplitter.split_text_calls == ["# Intro\n## Details\n" + long_markdown_body]
    assert _FakeRecursiveCharacterTextSplitter.init_args == [(100, 0)]
    assert _FakeRecursiveCharacterTextSplitter.split_calls == [["# Intro\n## Details\n" + long_markdown_body]]
    assert len(result.parent_chunks) > 1
    assert all(parent.metadata["header_1"] == "Intro" for parent in result.parent_chunks)
    assert all(parent.metadata["header_2"] == "Details" for parent in result.parent_chunks)
    assert all(parent.metadata["source"] == "guide.md" for parent in result.parent_chunks)


def test_auto_uses_recursive_for_md_and_txt_parents(monkeypatch):
    _reset_child_routing_trackers()
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(page_content="Markdown parent content", metadata={"file_type": ".md", "source": "guide.md"}),
            Document(page_content="Plain text parent content", metadata={"file_type": ".txt", "source": "notes.txt"}),
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "ignored"})],
        ChunkingConfig(child_splitter_mode="auto", parent_chunk_size=80, parent_chunk_overlap=0, child_chunk_size=20),
        _FakeEmbeddings(),
    )

    assert _TrackingSemanticChildChunker.init_args == []
    assert set(_flatten_split_calls(_TrackingRecursiveChildSplitter.split_calls)) == {
        "Markdown parent content",
        "Plain text parent content",
    }
    assert all(child.page_content.startswith("recursive-child:") for child in result.child_chunks)


def test_auto_uses_semantic_for_long_docx_parents(monkeypatch):
    _reset_child_routing_trackers()
    docx_text = "This is a long, continuous paragraph of natural language. " * 20
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(
                page_content=docx_text,
                metadata={"file_type": ".docx", "source": "chapter.docx"},
            )
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "ignored"})],
        ChunkingConfig(
            child_splitter_mode="auto",
            semantic_min_parent_length=200,
            semantic_max_line_density=0.2,
            parent_chunk_size=80,
            parent_chunk_overlap=0,
            child_chunk_size=20,
        ),
        _FakeEmbeddings(),
    )

    assert _TrackingSemanticChildChunker.split_calls == [[docx_text]]
    assert _TrackingRecursiveChildSplitter.init_args == []
    assert _TrackingRecursiveChildSplitter.split_calls == []
    assert result.child_chunks[0].page_content.startswith("semantic-child:")


def test_auto_uses_recursive_for_structured_pdf_parents(monkeypatch):
    _reset_child_routing_trackers()
    pdf_text = "Overview\n- item one\n- item two\nSection A\n1. step one\n2. step two"
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(
                page_content=pdf_text,
                metadata={"file_type": ".pdf", "source": "report.pdf"},
            )
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "ignored"})],
        ChunkingConfig(child_splitter_mode="auto", parent_chunk_size=80, parent_chunk_overlap=0, child_chunk_size=20),
        _FakeEmbeddings(),
    )

    assert _TrackingSemanticChildChunker.init_args == []
    assert _TrackingRecursiveChildSplitter.split_calls == [[pdf_text]]
    assert result.child_chunks[0].page_content.startswith("recursive-child:")


def test_auto_uses_recursive_for_chinese_structured_pdf_parents(monkeypatch):
    _reset_child_routing_trackers()
    pdf_text = (
        "（一）适用范围说明 " + ("这是较长的中文段落，用来确保文本长度足够且行密度较低。" * 8) + "\n"
        "（二）处理规则说明 " + ("这里继续补充连续中文内容，避免因为长度或格式过短影响路由判断。" * 8)
    )
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(
                page_content=pdf_text,
                metadata={"file_type": ".pdf", "source": "report-cn.pdf"},
            )
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "ignored"})],
        ChunkingConfig(
            child_splitter_mode="auto",
            semantic_min_parent_length=100,
            semantic_max_line_density=0.1,
            parent_chunk_size=80,
            parent_chunk_overlap=0,
            child_chunk_size=20,
        ),
        _FakeEmbeddings(),
    )

    assert _TrackingSemanticChildChunker.init_args == []
    assert _TrackingSemanticChildChunker.split_calls == []
    assert _TrackingRecursiveChildSplitter.split_calls == [[pdf_text]]
    assert result.child_chunks[0].page_content.startswith("recursive-child:")


def test_auto_uses_recursive_for_chinese_article_structured_pdf_parents(monkeypatch):
    _reset_child_routing_trackers()
    pdf_text = (
        "第一条 适用范围 " + ("这是法规正文中的连续中文内容，用来保证文本足够长且不是高行密度列表。" * 8) + "\n"
        "第二条 处理流程 " + ("这里继续补充完整段落，确保自动路由时唯一的阻断因素来自中文结构化编号。" * 8)
    )
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(
                page_content=pdf_text,
                metadata={"file_type": ".pdf", "source": "regulation.pdf"},
            )
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "ignored"})],
        ChunkingConfig(
            child_splitter_mode="auto",
            semantic_min_parent_length=100,
            semantic_max_line_density=0.1,
            parent_chunk_size=80,
            parent_chunk_overlap=0,
            child_chunk_size=20,
        ),
        _FakeEmbeddings(),
    )

    assert _TrackingSemanticChildChunker.init_args == []
    assert _TrackingSemanticChildChunker.split_calls == []
    assert _TrackingRecursiveChildSplitter.split_calls == [[pdf_text]]
    assert result.child_chunks[0].page_content.startswith("recursive-child:")


@pytest.mark.parametrize(
    ("title_prefix", "source"),
    [
        ("（1）适用范围说明", "report-cn-digit-paren.pdf"),
        ("1、适用范围说明", "report-cn-digit-comma.pdf"),
        ("第1条 适用范围", "regulation-digit.pdf"),
    ],
)
def test_auto_uses_recursive_for_arabic_number_structured_pdf_parents(monkeypatch, title_prefix, source):
    _reset_child_routing_trackers()
    pdf_text = (
        title_prefix + ("这是较长的中文段落，用来确保文本长度足够且行密度较低。" * 8) + "\n"
        "第二部分 处理流程 " + ("这里继续补充完整段落，确保自动路由只会被结构化编号命中。" * 8)
    )
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(
                page_content=pdf_text,
                metadata={"file_type": ".pdf", "source": source},
            )
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "ignored"})],
        ChunkingConfig(
            child_splitter_mode="auto",
            semantic_min_parent_length=100,
            semantic_max_line_density=0.1,
            parent_chunk_size=80,
            parent_chunk_overlap=0,
            child_chunk_size=20,
        ),
        _FakeEmbeddings(),
    )

    assert _TrackingSemanticChildChunker.init_args == []
    assert _TrackingSemanticChildChunker.split_calls == []
    assert _TrackingRecursiveChildSplitter.split_calls == [[pdf_text]]
    assert result.child_chunks[0].page_content.startswith("recursive-child:")


def test_forced_recursive_mode_uses_recursive_only(monkeypatch):
    _reset_child_routing_trackers()
    forced_recursive_text = "This is a long, continuous paragraph of natural language. " * 20
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(
                page_content=forced_recursive_text,
                metadata={"file_type": ".docx", "source": "forced.docx"},
            )
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "ignored"})],
        ChunkingConfig(child_splitter_mode="recursive", parent_chunk_size=80, parent_chunk_overlap=0, child_chunk_size=20),
        _FakeEmbeddings(),
    )

    assert _TrackingSemanticChildChunker.init_args == []
    assert _TrackingRecursiveChildSplitter.split_calls == [[forced_recursive_text]]
    assert result.child_chunks[0].page_content.startswith("recursive-child:")


def test_forced_semantic_mode_uses_semantic_even_for_txt_and_structured_text(monkeypatch):
    _reset_child_routing_trackers()
    semantic_text = "Title\n- item one\n- item two\nSection B\nAnother paragraph"
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(
                page_content=semantic_text,
                metadata={"file_type": ".txt", "source": "forced.txt"},
            )
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "ignored"})],
        ChunkingConfig(child_splitter_mode="semantic", parent_chunk_size=80, parent_chunk_overlap=0, child_chunk_size=20),
        _FakeEmbeddings(),
    )

    assert _TrackingSemanticChildChunker.split_calls == [[semantic_text]]
    assert _TrackingRecursiveChildSplitter.init_args == []
    assert result.child_chunks[0].page_content.startswith("semantic-child:")


def test_chunking_config_rejects_non_positive_semantic_chunk_workers():
    with pytest.raises(ValueError, match="semantic_chunk_workers"):
        ChunkingConfig(semantic_chunk_workers=0)

    with pytest.raises(ValueError, match="semantic_chunk_workers"):
        ChunkingConfig(semantic_chunk_workers=-1)


def test_chunking_config_rejects_invalid_parent_chunk_thresholds():
    with pytest.raises(ValueError, match="parent_chunk_size"):
        ChunkingConfig(parent_chunk_size=0)

    with pytest.raises(ValueError, match="parent_chunk_overlap"):
        ChunkingConfig(parent_chunk_overlap=-1)

    with pytest.raises(ValueError, match="parent_chunk_overlap"):
        ChunkingConfig(parent_chunk_size=120, parent_chunk_overlap=120)


def test_chunking_config_accepts_default_child_splitter_routing():
    config = ChunkingConfig()

    assert config.child_splitter_mode == "auto"
    assert config.child_recursive_chunk_size == 400
    assert config.child_recursive_chunk_overlap == 80
    assert config.semantic_file_types == (".pdf", ".docx")
    assert config.semantic_min_parent_length == 1000
    assert config.semantic_max_line_density == 0.03


def test_chunking_config_rejects_invalid_child_splitter_mode():
    with pytest.raises(ValueError, match="child_splitter_mode"):
        ChunkingConfig(child_splitter_mode="invalid")


def test_chunking_config_rejects_invalid_recursive_child_thresholds():
    with pytest.raises(ValueError, match="child_recursive_chunk_size"):
        ChunkingConfig(child_recursive_chunk_size=0)

    with pytest.raises(ValueError, match="child_recursive_chunk_overlap"):
        ChunkingConfig(child_recursive_chunk_overlap=-1)

    with pytest.raises(ValueError, match="child_recursive_chunk_overlap"):
        ChunkingConfig(child_recursive_chunk_size=80, child_recursive_chunk_overlap=80)


def test_chunking_config_rejects_invalid_semantic_thresholds():
    with pytest.raises(ValueError, match="semantic_min_parent_length"):
        ChunkingConfig(semantic_min_parent_length=0)

    with pytest.raises(ValueError, match="semantic_max_line_density"):
        ChunkingConfig(semantic_max_line_density=-0.01)

    with pytest.raises(ValueError, match="semantic_max_line_density"):
        ChunkingConfig(semantic_max_line_density=1.01)


def test_build_parent_child_chunks_uses_semantic_chunk_workers_from_config(monkeypatch):
    _RecordingExecutor.created_max_workers = []
    _RecordingExecutor.submitted_calls = []
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _FakeSemanticChunker)
    monkeypatch.setattr(chunking, "ThreadPoolExecutor", _RecordingExecutor, raising=False)

    documents = [
        Document(
            page_content="\u8fd9\u662f\u7528\u4e8e\u6d4b\u8bd5\u7236\u5b50\u5206\u5757\u7684\u4e2d\u6587\u5185\u5bb9\u3002",
            metadata={"page": 3, "source": "demo.pdf", "file_path": "demo.pdf", "loader_type": "digital"},
        )
    ]

    build_parent_child_chunks(
        documents,
        ChunkingConfig(parent_chunk_size=200, parent_chunk_overlap=0, child_chunk_size=20, semantic_chunk_workers=7),
        _FakeEmbeddings(),
    )

    assert _RecordingExecutor.created_max_workers == [7]


def test_build_parent_child_chunks_preserves_parent_and_child_order_with_parallel_processing(monkeypatch):
    _reset_child_routing_trackers()
    _OutOfOrderExecutor.created_max_workers = []
    _OutOfOrderExecutor.submitted_parent_ids = []
    _OutOfOrderExecutor.completed_parent_ids = []
    _TrackingRecursiveChildSplitter.init_args = []
    _TrackingRecursiveChildSplitter.split_calls = []
    monkeypatch.setattr(chunking, "_load_semantic_chunker_class", lambda: _TrackingSemanticChildChunker)
    monkeypatch.setattr(chunking, "ThreadPoolExecutor", _OutOfOrderExecutor, raising=False)
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", _TrackingRecursiveChildSplitter)
    monkeypatch.setattr(
        chunking,
        "_split_parent_documents",
        lambda documents, config: [
            Document(page_content="parent-0", metadata={"source": "demo.pdf", "file_type": ".pdf"}),
            Document(page_content="parent-1", metadata={"source": "demo.pdf", "file_type": ".pdf"}),
            Document(page_content="parent-2", metadata={"source": "demo.pdf", "file_type": ".pdf"}),
        ],
    )

    result = build_parent_child_chunks(
        [Document(page_content="ignored", metadata={"source": "demo.txt"})],
        ChunkingConfig(
            child_splitter_mode="auto",
            semantic_min_parent_length=1,
            semantic_max_line_density=0.2,
            parent_chunk_size=50,
            parent_chunk_overlap=0,
            child_chunk_size=20,
            semantic_chunk_workers=3,
        ),
        _FakeEmbeddings(),
    )

    assert _OutOfOrderExecutor.created_max_workers == [3]
    assert _OutOfOrderExecutor.submitted_parent_ids == ["parent-0", "parent-1", "parent-2"]
    assert _OutOfOrderExecutor.completed_parent_ids == ["parent-2", "parent-1", "parent-0"]
    assert _TrackingSemanticChildChunker.split_calls == [["parent-2"], ["parent-1"], ["parent-0"]]
    assert _TrackingRecursiveChildSplitter.init_args == []
    assert _TrackingRecursiveChildSplitter.split_calls == []
    assert [parent.metadata["parent_index"] for parent in result.parent_chunks] == [0, 1, 2]
    assert [parent.metadata["parent_id"] for parent in result.parent_chunks] == [
        "parent-0",
        "parent-1",
        "parent-2",
    ]
    assert [child.metadata["parent_id"] for child in result.child_chunks] == [
        "parent-0",
        "parent-1",
        "parent-2",
    ]
    assert [child.metadata["child_index"] for child in result.child_chunks] == [0, 0, 0]
    assert [child.metadata["child_id"] for child in result.child_chunks] == [
        "parent-0-child-0",
        "parent-1-child-0",
        "parent-2-child-0",
    ]
