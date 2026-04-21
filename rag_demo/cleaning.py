from __future__ import annotations

import re

from langchain_core.documents import Document


def _normalize_line(line: str) -> str:
    if not line.strip():
        return ""

    leading_whitespace = re.match(r"^[ \t]*", line).group(0)
    content = line[len(leading_whitespace) :]
    content = re.sub(r"[ \t]+", " ", content).rstrip()
    return f"{leading_whitespace}{content}"


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(_normalize_line(line) for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    lines = normalized.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def clean_documents(documents: list[Document]) -> list[Document]:
    cleaned_documents: list[Document] = []

    for document in documents:
        cleaned_text = _normalize_text(document.page_content)
        if not cleaned_text:
            continue
        cleaned_documents.append(
            Document(
                id=document.id,
                page_content=cleaned_text,
                metadata=dict(document.metadata),
            )
        )

    return cleaned_documents
