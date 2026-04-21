from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .models import ChunkingConfig, ChunkingResult

# Chinese sentence terminators: \u3002 \uff01 \uff1f
DEFAULT_SENTENCE_SPLIT_REGEX = r"(?<=[\u3002\uff01\uff1f.!?])\s*"
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
]
CHINESE_NUMBER_MARKER_CHARS = (
    "\u3007\u96f6"
    "\u4e00\u4e8c\u4e09\u56db\u4e94"
    "\u516d\u4e03\u516b\u4e5d\u5341"
    "\u767e\u5343\u4e07"
)
STRUCTURAL_MARKER_PATTERNS = (
    re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE),
    re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE),
    re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE),
    re.compile(r"^\s*(?:[\(\uFF08]\d+[\)\uFF09]|\d+[\u3001.\uFF0E])\s*\S", re.MULTILINE),
    re.compile(
        rf"^\s*(?:[\(\uFF08][{CHINESE_NUMBER_MARKER_CHARS}]+[\)\uFF09]|"
        rf"[{CHINESE_NUMBER_MARKER_CHARS}]+[\u3001.\uFF0E])\s*\S",
        re.MULTILINE,
    ),
    re.compile(r"^\s*\u7b2c\d+\s*[\u7ae0\u8282\u6761\u6b3e\u9879]\s*\S", re.MULTILINE),
    re.compile(
        rf"^\s*\u7b2c[{CHINESE_NUMBER_MARKER_CHARS}]+\s*[\u7ae0\u8282\u6761\u6b3e\u9879]\s*\S",
        re.MULTILINE,
    ),
)


def _parent_document(document: Document, parent_id: str, parent_index: int) -> Document:
    metadata = dict(document.metadata)
    metadata["parent_id"] = parent_id
    metadata["parent_index"] = parent_index
    return Document(page_content=document.page_content, metadata=metadata, id=document.id)


def _child_document(parent: Document, child: Document, child_index: int) -> Document:
    metadata = dict(parent.metadata)
    metadata.update(child.metadata)
    metadata["parent_id"] = parent.metadata["parent_id"]
    metadata["parent_index"] = parent.metadata["parent_index"]
    metadata["child_id"] = f"{parent.metadata['parent_id']}-child-{child_index}"
    metadata["child_index"] = child_index
    metadata["parent_text"] = parent.page_content
    return Document(page_content=child.page_content, metadata=metadata, id=child.id)


def _load_semantic_chunker_class():
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError as exc:
        raise ImportError("langchain-experimental is required for semantic chunking") from exc

    return SemanticChunker


def _build_semantic_splitter(embeddings, config: ChunkingConfig):
    try:
        semantic_chunker_class = _load_semantic_chunker_class()
    except ImportError as exc:
        raise RuntimeError(
            "SemanticChunker requires langchain-experimental to be installed."
        ) from exc

    return semantic_chunker_class(
        embeddings,
        sentence_split_regex=DEFAULT_SENTENCE_SPLIT_REGEX,
        min_chunk_size=config.child_chunk_size,
    )


def _build_recursive_child_splitter(config: ChunkingConfig):
    return RecursiveCharacterTextSplitter(
        chunk_size=config.child_recursive_chunk_size,
        chunk_overlap=config.child_recursive_chunk_overlap,
    )


def _recursive_parent_documents(
    documents: list[Document],
    config: ChunkingConfig,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.parent_chunk_size,
        chunk_overlap=config.parent_chunk_overlap,
    )
    return splitter.split_documents(documents)


def _split_markdown_parent_section(
    document: Document,
    config: ChunkingConfig,
) -> list[Document]:
    if len(document.page_content) <= config.parent_chunk_size:
        return [document]

    return _recursive_parent_documents([document], config)


def _markdown_parent_documents(
    document: Document,
    config: ChunkingConfig,
) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    header_documents = splitter.split_text(document.page_content)
    if not header_documents:
        return _split_markdown_parent_section(document, config)

    parent_chunks: list[Document] = []

    for header_document in header_documents:
        section_document = Document(
            page_content=header_document.page_content,
            metadata={**dict(document.metadata), **dict(header_document.metadata)},
            id=document.id,
        )
        parent_chunks.extend(_split_markdown_parent_section(section_document, config))

    return parent_chunks


def _split_parent_documents(
    documents: list[Document],
    config: ChunkingConfig,
) -> list[Document]:
    parent_chunks: list[Document] = []

    for document in documents:
        file_type = str(document.metadata.get("file_type", "")).lower()
        if file_type == ".md":
            parent_chunks.extend(_markdown_parent_documents(document, config))
            continue

        parent_chunks.extend(_recursive_parent_documents([document], config))

    return parent_chunks


def _line_density(text: str) -> float:
    if not text:
        return 0.0

    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    return len(non_empty_lines) / len(text)


def _has_structural_markers(text: str) -> bool:
    return any(pattern.search(text) for pattern in STRUCTURAL_MARKER_PATTERNS)


def _should_use_semantic_for_parent(parent: Document, config: ChunkingConfig) -> bool:
    file_type = str(parent.metadata.get("file_type", "")).lower()
    text = parent.page_content

    if file_type not in config.semantic_file_types:
        return False
    if len(text) < config.semantic_min_parent_length:
        return False
    if _line_density(text) >= config.semantic_max_line_density:
        return False
    if _has_structural_markers(text):
        return False

    return True


def _split_child_documents_with_recursive(parent: Document, recursive_splitter) -> list[Document]:
    recursive_children = recursive_splitter.split_documents([parent])
    if not recursive_children:
        recursive_children = [parent]

    return [_child_document(parent, child, child_index) for child_index, child in enumerate(recursive_children)]


def _split_child_documents_with_semantic(parent: Document, semantic_splitter) -> list[Document]:
    semantic_children = semantic_splitter.split_documents([parent])
    if not semantic_children:
        semantic_children = [parent]

    return [_child_document(parent, child, child_index) for child_index, child in enumerate(semantic_children)]


def _child_splitter_mode_for_parent(parent: Document, config: ChunkingConfig) -> str:
    if config.child_splitter_mode == "semantic":
        return "semantic"
    if config.child_splitter_mode == "recursive":
        return "recursive"
    if _should_use_semantic_for_parent(parent, config):
        return "semantic"
    return "recursive"


def _split_child_documents(
    parent: Document,
    child_splitter_mode: str,
    *,
    semantic_splitter=None,
    recursive_splitter=None,
) -> list[Document]:
    if child_splitter_mode == "semantic":
        if semantic_splitter is None:
            raise ValueError("semantic_splitter is required for semantic child splitting")
        return _split_child_documents_with_semantic(parent, semantic_splitter)

    if recursive_splitter is None:
        raise ValueError("recursive_splitter is required for recursive child splitting")
    return _split_child_documents_with_recursive(parent, recursive_splitter)


def build_parent_child_chunks(
    documents: list[Document],
    config: ChunkingConfig,
    embeddings,
) -> ChunkingResult:
    parent_chunks: list[Document] = []
    child_chunks: list[Document] = []

    for parent_index, parent in enumerate(_split_parent_documents(documents, config)):
        parent_id = f"parent-{parent_index}"
        parent_document = _parent_document(parent, parent_id, parent_index)
        parent_chunks.append(parent_document)

    child_splitter_modes = [_child_splitter_mode_for_parent(parent_document, config) for parent_document in parent_chunks]
    semantic_splitter = (
        _build_semantic_splitter(embeddings, config)
        if "semantic" in child_splitter_modes
        else None
    )
    recursive_splitter = (
        _build_recursive_child_splitter(config)
        if "recursive" in child_splitter_modes
        else None
    )

    with ThreadPoolExecutor(max_workers=config.semantic_chunk_workers) as executor:
        futures = [
            executor.submit(
                _split_child_documents,
                parent_document,
                child_splitter_modes[parent_index],
                semantic_splitter=semantic_splitter,
                recursive_splitter=recursive_splitter,
            )
            for parent_index, parent_document in enumerate(parent_chunks)
        ]

        for future in futures:
            parent_children = future.result()
            child_chunks.extend(parent_children)

    return ChunkingResult(parent_chunks=parent_chunks, child_chunks=child_chunks)
