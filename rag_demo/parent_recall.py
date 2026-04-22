from __future__ import annotations

from langchain_core.documents import Document


def _rerank_score(candidate: Document) -> float:
    return float(candidate.metadata.get("rerank_score", 0.0))


def collapse_to_parent_hits(reranked_candidates: list[Document], limit: int = 5) -> list[Document]:
    best_by_parent: dict[str, tuple[float, int, Document]] = {}

    for index, candidate in enumerate(reranked_candidates):
        parent_id = candidate.metadata.get("parent_id")
        if parent_id is None:
            continue

        score = _rerank_score(candidate)
        current = best_by_parent.get(str(parent_id))
        if current is None or score > current[0]:
            best_by_parent[str(parent_id)] = (
                score,
                index,
                Document(page_content=candidate.page_content, metadata=dict(candidate.metadata), id=candidate.id),
            )

    collapsed = sorted(best_by_parent.values(), key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in collapsed[:limit]]


def _parent_chunks_collection(mongo_repository):
    collection = getattr(mongo_repository, "_parent_chunks", None)
    if collection is not None:
        return collection

    client = getattr(mongo_repository, "client", None)
    database_name = getattr(mongo_repository, "database_name", None)
    if client is None or database_name is None:
        raise AttributeError("mongo_repository must expose _parent_chunks or client/database_name")

    return client[database_name]["parent_chunks"]


def _parent_chunk_document(record: dict) -> Document:
    metadata = dict(record.get("metadata") or {})
    parent_id = record.get("parent_id", metadata.get("parent_id"))
    if parent_id is not None:
        metadata["parent_id"] = parent_id
    if "parent_index" not in metadata and record.get("parent_index") is not None:
        metadata["parent_index"] = record["parent_index"]
    return Document(page_content=str(record.get("text", "")), metadata=metadata)


def fetch_parent_chunks(parent_ids: list[str], mongo_repository) -> list[Document]:
    if not parent_ids:
        return []

    collection = _parent_chunks_collection(mongo_repository)
    unique_parent_ids = list(dict.fromkeys(str(parent_id) for parent_id in parent_ids))
    records = list(collection.find({"parent_id": {"$in": unique_parent_ids}}))

    records_by_parent_id: dict[str, dict] = {}
    for record in records:
        parent_id = record.get("parent_id")
        if parent_id is not None and str(parent_id) not in records_by_parent_id:
            records_by_parent_id[str(parent_id)] = dict(record)

    ordered_chunks: list[Document] = []
    for parent_id in parent_ids:
        record = records_by_parent_id.get(str(parent_id))
        if record is not None:
            ordered_chunks.append(_parent_chunk_document(record))

    return ordered_chunks


async def fetch_parent_chunks_async(parent_ids: list[str], mongo_repository) -> list[Document]:
    if not parent_ids:
        return []

    collection = getattr(mongo_repository, "_async_parent_chunks", None)
    if collection is None:
        raise AttributeError("mongo_repository must expose _async_parent_chunks for async fetch")

    unique_parent_ids = list(dict.fromkeys(str(parent_id) for parent_id in parent_ids))
    cursor = collection.find({"parent_id": {"$in": unique_parent_ids}})
    records = await cursor.to_list(length=None)

    records_by_parent_id: dict[str, dict] = {}
    for record in records:
        parent_id = record.get("parent_id")
        if parent_id is not None and str(parent_id) not in records_by_parent_id:
            records_by_parent_id[str(parent_id)] = dict(record)

    ordered_chunks: list[Document] = []
    for parent_id in parent_ids:
        record = records_by_parent_id.get(str(parent_id))
        if record is not None:
            ordered_chunks.append(_parent_chunk_document(record))

    return ordered_chunks


__all__ = ["collapse_to_parent_hits", "fetch_parent_chunks", "fetch_parent_chunks_async"]
