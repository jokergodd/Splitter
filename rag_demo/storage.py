from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from pymongo.asynchronous.mongo_client import AsyncMongoClient
from pymongo import MongoClient
from qdrant_client import AsyncQdrantClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def compute_content_hash(file_path: str | Path) -> str:
    path = Path(file_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_child_chunk_point_id(content_hash: str, child_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{content_hash}:{child_id}"))


def _build_child_chunk_payload(content_hash: str, child_chunk: Document) -> dict[str, object]:
    child_id = str(child_chunk.metadata["child_id"])
    return {
        "content_hash": content_hash,
        "file_type": child_chunk.metadata.get("file_type"),
        "file_path": child_chunk.metadata.get("file_path"),
        "source": child_chunk.metadata.get("source"),
        "parent_id": child_chunk.metadata.get("parent_id"),
        "parent_index": child_chunk.metadata.get("parent_index"),
        "child_id": child_id,
        "child_index": child_chunk.metadata.get("child_index"),
        "text": child_chunk.page_content,
    }


def _coerce_sparse_document_vector(vector) -> dict[str, list[float] | list[int]]:
    sparse_object = vector.as_object() if hasattr(vector, "as_object") else vector
    indices = sparse_object["indices"]
    values = sparse_object["values"]
    return {
        "indices": list(indices),
        "values": list(values),
    }


def _embed_sparse_documents(sparse_embeddings, texts: list[str]) -> list[dict[str, list[float] | list[int]]]:
    if hasattr(sparse_embeddings, "embed_documents"):
        return [_coerce_sparse_document_vector(vector) for vector in sparse_embeddings.embed_documents(texts)]
    if hasattr(sparse_embeddings, "passage_embed"):
        return [_coerce_sparse_document_vector(vector) for vector in sparse_embeddings.passage_embed(texts)]
    if hasattr(sparse_embeddings, "embed"):
        return [_coerce_sparse_document_vector(vector) for vector in sparse_embeddings.embed(texts)]
    raise TypeError(
        "sparse_embeddings must expose embed_documents(...), passage_embed(...), or embed(...)"
    )


class MongoIngestionRepository:
    def __init__(self, client, database_name: str = "splitter", *, async_client=None):
        self.client = client
        self.async_client = async_client
        self.database_name = database_name
        self._database = self.client[self.database_name]
        self._ingested_files = self._database["ingested_files"]
        self._parent_chunks = self._database["parent_chunks"]
        self._async_database = None if self.async_client is None else self.async_client[self.database_name]
        self._async_ingested_files = (
            None if self._async_database is None else self._async_database["ingested_files"]
        )
        self._async_parent_chunks = (
            None if self._async_database is None else self._async_database["parent_chunks"]
        )

    def _find_file_record(self, content_hash: str):
        return self._ingested_files.find_one({"content_hash": content_hash})

    def should_skip_hash(self, content_hash: str) -> bool:
        record = self._find_file_record(content_hash)
        return bool(record and record.get("status") == "completed")

    async def should_skip_hash_async(self, content_hash: str) -> bool:
        if self._async_ingested_files is None:
            raise AttributeError("async ingested files collection is not configured")
        record = await self._async_ingested_files.find_one({"content_hash": content_hash})
        return bool(record and record.get("status") == "completed")

    def mark_processing(
        self,
        *,
        content_hash: str,
        file_path: Path,
        file_type: str,
        file_size: int,
    ) -> None:
        now = _utcnow()
        self._ingested_files.update_one(
            {"content_hash": content_hash},
            {
                "$set": {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_type,
                    "file_size": file_size,
                    "status": "processing",
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "content_hash": content_hash,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    async def mark_processing_async(
        self,
        *,
        content_hash: str,
        file_path: Path,
        file_type: str,
        file_size: int,
    ) -> None:
        if self._async_ingested_files is None:
            raise AttributeError("async ingested files collection is not configured")
        now = _utcnow()
        await self._async_ingested_files.update_one(
            {"content_hash": content_hash},
            {
                "$set": {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_type,
                    "file_size": file_size,
                    "status": "processing",
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "content_hash": content_hash,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    def mark_completed(
        self,
        *,
        content_hash: str,
        raw_page_count: int,
        cleaned_page_count: int,
        parent_chunk_count: int,
        child_chunk_count: int,
        parent_ids: list[str],
    ) -> None:
        self._ingested_files.update_one(
            {"content_hash": content_hash},
            {
                "$set": {
                    "status": "completed",
                    "raw_page_count": raw_page_count,
                    "cleaned_page_count": cleaned_page_count,
                    "parent_chunk_count": parent_chunk_count,
                    "child_chunk_count": child_chunk_count,
                    "parent_ids": parent_ids,
                    "error": None,
                    "updated_at": _utcnow(),
                }
            },
        )

    async def mark_completed_async(
        self,
        *,
        content_hash: str,
        raw_page_count: int,
        cleaned_page_count: int,
        parent_chunk_count: int,
        child_chunk_count: int,
        parent_ids: list[str],
    ) -> None:
        if self._async_ingested_files is None:
            raise AttributeError("async ingested files collection is not configured")
        await self._async_ingested_files.update_one(
            {"content_hash": content_hash},
            {
                "$set": {
                    "status": "completed",
                    "raw_page_count": raw_page_count,
                    "cleaned_page_count": cleaned_page_count,
                    "parent_chunk_count": parent_chunk_count,
                    "child_chunk_count": child_chunk_count,
                    "parent_ids": parent_ids,
                    "error": None,
                    "updated_at": _utcnow(),
                }
            },
        )

    def mark_failed(self, *, content_hash: str, error: str) -> None:
        self._ingested_files.update_one(
            {"content_hash": content_hash},
            {
                "$set": {
                    "status": "failed",
                    "error": error,
                    "updated_at": _utcnow(),
                }
            },
        )

    async def mark_failed_async(self, *, content_hash: str, error: str) -> None:
        if self._async_ingested_files is None:
            raise AttributeError("async ingested files collection is not configured")
        await self._async_ingested_files.update_one(
            {"content_hash": content_hash},
            {
                "$set": {
                    "status": "failed",
                    "error": error,
                    "updated_at": _utcnow(),
                }
            },
        )

    def store_parent_chunks(
        self,
        *,
        content_hash: str,
        file_type: str,
        parent_chunks: list[Document],
    ) -> list[str]:
        parent_ids: list[str] = []
        created_at = _utcnow()

        for parent in parent_chunks:
            parent_id = str(parent.metadata["parent_id"])
            parent_ids.append(parent_id)
            self._parent_chunks.update_one(
                {
                    "content_hash": content_hash,
                    "parent_id": parent_id,
                },
                {
                    "$set": {
                        "file_type": file_type,
                        "parent_index": parent.metadata.get("parent_index"),
                        "text": parent.page_content,
                        "metadata": dict(parent.metadata),
                    },
                    "$setOnInsert": {
                        "content_hash": content_hash,
                        "parent_id": parent_id,
                        "created_at": created_at,
                    },
                },
                upsert=True,
            )

        return parent_ids

    async def store_parent_chunks_async(
        self,
        *,
        content_hash: str,
        file_type: str,
        parent_chunks: list[Document],
    ) -> list[str]:
        if self._async_parent_chunks is None:
            raise AttributeError("async parent chunks collection is not configured")

        parent_ids: list[str] = []
        created_at = _utcnow()

        for parent in parent_chunks:
            parent_id = str(parent.metadata["parent_id"])
            parent_ids.append(parent_id)
            await self._async_parent_chunks.update_one(
                {
                    "content_hash": content_hash,
                    "parent_id": parent_id,
                },
                {
                    "$set": {
                        "file_type": file_type,
                        "parent_index": parent.metadata.get("parent_index"),
                        "text": parent.page_content,
                        "metadata": dict(parent.metadata),
                    },
                    "$setOnInsert": {
                        "content_hash": content_hash,
                        "parent_id": parent_id,
                        "created_at": created_at,
                    },
                },
                upsert=True,
            )

        return parent_ids


class QdrantHybridChildStore:
    def __init__(
        self,
        client,
        collection_name: str = "child_chunks_hybrid",
        *,
        async_client=None,
        sparse_embeddings=None,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
    ):
        self.client = client
        self.async_client = async_client
        self.collection_name = collection_name
        self.sparse_embeddings = sparse_embeddings
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name

    def _build_points(
        self,
        *,
        content_hash: str,
        child_chunks: list[Document],
        embeddings,
        sparse_embeddings=None,
    ) -> tuple[int, list[dict]]:
        child_texts = [child_chunk.page_content for child_chunk in child_chunks]
        dense_vectors = embeddings.embed_documents(child_texts)
        if len(dense_vectors) != len(child_chunks):
            raise ValueError(
                "dense vectors count must match child_chunks count for hybrid child storage"
            )
        sparse_encoder = sparse_embeddings or self.sparse_embeddings
        if sparse_encoder is None:
            raise ValueError("sparse_embeddings must be provided for hybrid child chunk storage")
        sparse_vectors = _embed_sparse_documents(sparse_encoder, child_texts)
        if len(sparse_vectors) != len(child_chunks):
            raise ValueError(
                "sparse vectors count must match child_chunks count for hybrid child storage"
            )
        points = []

        for child_chunk, dense_vector, sparse_vector in zip(
            child_chunks,
            dense_vectors,
            sparse_vectors,
        ):
            child_id = str(child_chunk.metadata["child_id"])
            points.append(
                {
                    "id": _build_child_chunk_point_id(content_hash, child_id),
                    "vector": {
                        self.dense_vector_name: dense_vector,
                        self.sparse_vector_name: sparse_vector,
                    },
                    "payload": _build_child_chunk_payload(content_hash, child_chunk),
                }
            )

        return len(dense_vectors[0]), points

    def store_child_chunks(
        self,
        *,
        content_hash: str,
        child_chunks: list[Document],
        embeddings,
        sparse_embeddings=None,
    ) -> int:
        if not child_chunks:
            return 0

        vector_size, points = self._build_points(
            content_hash=content_hash,
            child_chunks=child_chunks,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
        )
        self._ensure_collection(vector_size=vector_size)
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    async def store_child_chunks_async(
        self,
        *,
        content_hash: str,
        child_chunks: list[Document],
        embeddings,
        sparse_embeddings=None,
    ) -> int:
        if not child_chunks:
            return 0
        if self.async_client is None:
            raise AttributeError("async qdrant client is not configured")

        vector_size, points = await asyncio.to_thread(
            self._build_points,
            content_hash=content_hash,
            child_chunks=child_chunks,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
        )
        await self._ensure_collection_async(vector_size=vector_size)
        await self.async_client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.dense_vector_name: VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.sparse_vector_name: SparseVectorParams(),
            },
        )

    async def _ensure_collection_async(self, vector_size: int) -> None:
        if self.async_client is None:
            raise AttributeError("async qdrant client is not configured")
        if await self.async_client.collection_exists(self.collection_name):
            return
        await self.async_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.dense_vector_name: VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.sparse_vector_name: SparseVectorParams(),
            },
        )


@dataclass(slots=True)
class StorageBackend:
    mongo_repository: MongoIngestionRepository
    qdrant_store: QdrantHybridChildStore
    sparse_embeddings: object


def build_storage_backend(
    mongo_uri: str = "mongodb://admin:123456@localhost:27017",
    mongo_database: str = "splitter",
    qdrant_url: str = "http://localhost:6333",
    qdrant_collection: str = "child_chunks_hybrid",
    *,
    sparse_embeddings,
) -> StorageBackend:
    mongo_client = MongoClient(mongo_uri)
    async_mongo_client = AsyncMongoClient(mongo_uri)
    qdrant_client = QdrantClient(
        url=qdrant_url,
        check_compatibility=False,
        trust_env=False,
    )
    async_qdrant_client = AsyncQdrantClient(
        url=qdrant_url,
        check_compatibility=False,
        trust_env=False,
    )
    return StorageBackend(
        mongo_repository=MongoIngestionRepository(
            client=mongo_client,
            database_name=mongo_database,
            async_client=async_mongo_client,
        ),
        qdrant_store=QdrantHybridChildStore(
            client=qdrant_client,
            collection_name=qdrant_collection,
            async_client=async_qdrant_client,
            sparse_embeddings=sparse_embeddings,
        ),
        sparse_embeddings=sparse_embeddings,
    )
