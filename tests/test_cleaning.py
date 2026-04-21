from langchain_core.documents import Document

from rag_demo.cleaning import clean_documents


def test_clean_documents_filters_empty_pages_and_normalizes_whitespace():
    docs = [
        Document(page_content="  \n\t ", metadata={"page": 1, "source": "demo.pdf"}),
        Document(
            page_content="Title\r\n\r\n\r\nParagraph\t with   extra spaces.  ",
            metadata={"page": 2, "source": "demo.pdf"},
        ),
    ]

    cleaned = clean_documents(docs)

    assert len(cleaned) == 1
    assert cleaned[0].metadata["page"] == 2
    assert cleaned[0].page_content == "Title\n\nParagraph with extra spaces."


def test_clean_documents_preserves_leading_indentation():
    docs = [
        Document(
            page_content="    Indented\tline   with   spacing  ",
            metadata={"page": 3, "source": "demo.pdf"},
        )
    ]

    cleaned = clean_documents(docs)

    assert len(cleaned) == 1
    assert cleaned[0].page_content == "    Indented line with spacing"


def test_clean_documents_preserves_document_id():
    docs = [
        Document(
            id="doc-123",
            page_content="Title\r\n\r\nParagraph\t with   extra spaces.",
            metadata={"page": 2, "source": "demo.pdf"},
        )
    ]

    cleaned = clean_documents(docs)

    assert len(cleaned) == 1
    assert cleaned[0].id == "doc-123"
    assert cleaned[0].page_content == "Title\n\nParagraph with extra spaces."
