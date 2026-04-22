from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_demo import storage


class _FakeCollection:
    def __init__(self):
        self.documents: list[dict] = []
        self.operations: list[tuple[str, dict, dict | list[dict]]] = []

    def find_one(self, query: dict):
        self.operations.append(("find_one", query, {}))
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                return dict(document)
        return None

    def update_one(self, query: dict, update: dict, upsert: bool = False):
        self.operations.append(("update_one", query, update))
        target = None
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                target = document
                break

        if target is None and upsert:
            target = dict(query)
            self.documents.append(target)

        if target is not None:
            for operator, payload in update.items():
                if operator == "$set":
                    target.update(payload)
                elif operator == "$setOnInsert":
                    for key, value in payload.items():
                        target.setdefault(key, value)

    def insert_many(self, documents: list[dict], ordered: bool = True):
        self.operations.append(("insert_many", {"ordered": ordered}, documents))
        self.documents.extend(dict(document) for document in documents)


class _FakeDatabase:
    def __init__(self):
        self.collections: dict[str, _FakeCollection] = {}

    def __getitem__(self, name: str) -> _FakeCollection:
        if name not in self.collections:
            self.collections[name] = _FakeCollection()
        return self.collections[name]


class _FakeMongoClient:
    def __init__(self):
        self.databases: dict[str, _FakeDatabase] = {}

    def __getitem__(self, name: str) -> _FakeDatabase:
        if name not in self.databases:
            self.databases[name] = _FakeDatabase()
        return self.databases[name]


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


class _TrackingEmbeddings:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return [[float(len(text)), 99.0 + float(index)] for index, text in enumerate(texts)]


class _HybridEmbeddings:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


class _FakeSparseEncoder:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def embed_documents(self, texts: list[str]):
        self.calls.append(tuple(texts))
        return [
            {
                "indices": [0, 3],
                "values": [1.0, 0.25 + float(index)],
            }
            for index, text in enumerate(texts)
        ]


class _FastEmbedSparseVector:
    def __init__(self, *, indices: list[int], values: list[float]):
        self._payload = {"indices": indices, "values": values}

    def as_object(self):
        return dict(self._payload)


class _FastEmbedStyleSparseEncoder:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def passage_embed(self, texts: list[str]):
        self.calls.append(tuple(texts))
        for index, _ in enumerate(texts):
            yield _FastEmbedSparseVector(indices=[10, 20], values=[1.0, 0.5 + float(index)])


class _DenseCountMismatchEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index)] for index, text in enumerate(texts[:-1])]


class _SparseCountMismatchEncoder:
    def embed_documents(self, texts: list[str]):
        return [
            {
                "indices": [0, 3],
                "values": [1.0, 0.25 + float(index)],
            }
            for index, text in enumerate(texts[:-1])
        ]


class _FakeQdrantClient:
    def __init__(self):
        self.upserts: list[dict] = []
        self.created_collections: list[dict] = []
        self.existing_collections: set[str] = set()

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.existing_collections

    def create_collection(self, collection_name: str, vectors_config, **kwargs):
        self.created_collections.append(
            {
                "collection_name": collection_name,
                "vectors_config": vectors_config,
                **kwargs,
            }
        )
        self.existing_collections.add(collection_name)

    def upsert(self, *, collection_name: str, points: list[dict]):
        self.upserts.append(
            {
                "collection_name": collection_name,
                "points": points,
            }
        )


def test_compute_content_hash_uses_file_bytes(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_bytes(b"hello world")

    content_hash = storage.compute_content_hash(file_path)

    assert content_hash == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_should_skip_when_completed_hash_exists():
    client = _FakeMongoClient()
    client["splitter"]["ingested_files"].documents.append(
        {"content_hash": "abc", "status": "completed"}
    )
    repo = storage.MongoIngestionRepository(client=client, database_name="splitter")

    assert repo.should_skip_hash("abc") is True


def test_should_not_skip_when_failed_hash_exists():
    client = _FakeMongoClient()
    client["splitter"]["ingested_files"].documents.append(
        {"content_hash": "abc", "status": "failed"}
    )
    repo = storage.MongoIngestionRepository(client=client, database_name="splitter")

    assert repo.should_skip_hash("abc") is False


def test_mark_processing_upserts_file_record(tmp_path):
    client = _FakeMongoClient()
    repo = storage.MongoIngestionRepository(client=client, database_name="splitter")
    file_path = tmp_path / "demo.txt"
    file_path.write_text("hello", encoding="utf-8")

    repo.mark_processing(
        content_hash="hash-1",
        file_path=file_path,
        file_type=".txt",
        file_size=file_path.stat().st_size,
    )

    record = client["splitter"]["ingested_files"].documents[0]
    assert record["content_hash"] == "hash-1"
    assert record["status"] == "processing"
    assert record["file_name"] == "demo.txt"
    assert record["file_path"] == str(file_path)
    assert record["file_type"] == ".txt"


def test_store_parent_chunks_writes_parent_documents():
    client = _FakeMongoClient()
    repo = storage.MongoIngestionRepository(client=client, database_name="splitter")
    parents = [
        Document(
            page_content="parent text",
            metadata={"parent_id": "parent-1", "parent_index": 0, "source": "demo.txt"},
        )
    ]

    parent_ids = repo.store_parent_chunks(
        content_hash="hash-1",
        file_type=".txt",
        parent_chunks=parents,
    )

    stored = client["splitter"]["parent_chunks"].documents[0]
    assert parent_ids == ["parent-1"]
    assert stored["content_hash"] == "hash-1"
    assert stored["file_type"] == ".txt"
    assert stored["parent_id"] == "parent-1"
    assert stored["text"] == "parent text"


def test_store_parent_chunks_is_idempotent_for_retrying_same_parent():
    client = _FakeMongoClient()
    repo = storage.MongoIngestionRepository(client=client, database_name="splitter")
    parents = [
        Document(
            page_content="parent text",
            metadata={"parent_id": "parent-1", "parent_index": 0, "source": "demo.txt"},
        )
    ]

    first_ids = repo.store_parent_chunks(
        content_hash="hash-1",
        file_type=".txt",
        parent_chunks=parents,
    )
    second_ids = repo.store_parent_chunks(
        content_hash="hash-1",
        file_type=".txt",
        parent_chunks=parents,
    )

    stored_documents = client["splitter"]["parent_chunks"].documents
    assert first_ids == ["parent-1"]
    assert second_ids == ["parent-1"]
    assert len(stored_documents) == 1
    assert stored_documents[0]["parent_id"] == "parent-1"
    assert stored_documents[0]["content_hash"] == "hash-1"


def test_store_child_chunks_writes_qdrant_payloads():
    client = _FakeQdrantClient()
    store = storage.QdrantHybridChildStore(client=client, collection_name="child_chunks_hybrid")
    children = [
        Document(
            page_content="child text",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        )
    ]

    stored_count = store.store_child_chunks(
        content_hash="hash-1",
        child_chunks=children,
        embeddings=_FakeEmbeddings(),
        sparse_embeddings=_FakeSparseEncoder(),
    )

    assert stored_count == 1
    upsert = client.upserts[0]
    assert upsert["collection_name"] == "child_chunks_hybrid"
    point = upsert["points"][0]
    assert isinstance(uuid.UUID(point["id"]), uuid.UUID)
    assert point["payload"]["content_hash"] == "hash-1"
    assert point["payload"]["child_id"] == "parent-1-child-0"
    assert point["payload"]["parent_id"] == "parent-1"
    assert point["payload"]["text"] == "child text"
    assert point["vector"]["dense"] == [10.0, 0.0]
    assert point["vector"]["sparse"] == {"indices": [0, 3], "values": [1.0, 0.25]}


def test_store_child_chunks_creates_collection_when_missing():
    client = _FakeQdrantClient()
    store = storage.QdrantHybridChildStore(client=client, collection_name="child_chunks_hybrid")
    children = [
        Document(
            page_content="child text",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        )
    ]

    store.store_child_chunks(
        content_hash="hash-1",
        child_chunks=children,
        embeddings=_FakeEmbeddings(),
        sparse_embeddings=_FakeSparseEncoder(),
    )

    assert len(client.created_collections) == 1
    created = client.created_collections[0]
    assert created["collection_name"] == "child_chunks_hybrid"
    assert created["vectors_config"]["dense"].size == 2
    assert created["sparse_vectors_config"] is not None


def test_store_child_chunks_uses_the_passed_embedding_wrapper_for_vectors():
    client = _FakeQdrantClient()
    store = storage.QdrantHybridChildStore(client=client, collection_name="child_chunks_hybrid")
    embeddings = _TrackingEmbeddings()
    sparse_embeddings = _FakeSparseEncoder()
    children = [
        Document(
            page_content="child text",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        )
    ]

    store.store_child_chunks(
        content_hash="hash-1",
        child_chunks=children,
        embeddings=embeddings,
        sparse_embeddings=sparse_embeddings,
    )

    assert embeddings.calls == [("child text",)]
    assert sparse_embeddings.calls == [("child text",)]
    assert client.upserts[0]["points"][0]["vector"]["dense"] == [10.0, 99.0]


def test_store_child_chunks_raises_when_dense_vector_count_mismatches_child_chunks():
    client = _FakeQdrantClient()
    store = storage.QdrantHybridChildStore(client=client, collection_name="child_chunks_hybrid")
    children = [
        Document(
            page_content="child text 1",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        ),
        Document(
            page_content="child text 2",
            metadata={
                "child_id": "parent-1-child-1",
                "child_index": 1,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        ),
    ]

    with pytest.raises(ValueError, match="dense vectors"):
        store.store_child_chunks(
            content_hash="hash-1",
            child_chunks=children,
            embeddings=_DenseCountMismatchEmbeddings(),
            sparse_embeddings=_FakeSparseEncoder(),
        )


def test_store_child_chunks_uses_store_level_sparse_embeddings_when_argument_is_omitted():
    client = _FakeQdrantClient()
    sparse_embeddings = _FakeSparseEncoder()
    store = storage.QdrantHybridChildStore(
        client=client,
        collection_name="child_chunks_hybrid",
        sparse_embeddings=sparse_embeddings,
    )
    dense_embeddings = _HybridEmbeddings()
    children = [
        Document(
            page_content="child text",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        )
    ]

    stored_count = store.store_child_chunks(
        content_hash="hash-1",
        child_chunks=children,
        embeddings=dense_embeddings,
    )

    assert stored_count == 1
    assert dense_embeddings.calls == [("child text",)]
    assert sparse_embeddings.calls == [("child text",)]
    point = client.upserts[0]["points"][0]
    assert client.upserts[0]["collection_name"] == "child_chunks_hybrid"
    assert point["vector"]["dense"] == [10.0, 0.0]
    assert point["vector"]["sparse"] == {"indices": [0, 3], "values": [1.0, 0.25]}


def test_store_child_chunks_supports_fastembed_style_sparse_encoders():
    client = _FakeQdrantClient()
    sparse_embeddings = _FastEmbedStyleSparseEncoder()
    store = storage.QdrantHybridChildStore(
        client=client,
        collection_name="child_chunks_hybrid",
        sparse_embeddings=sparse_embeddings,
    )
    children = [
        Document(
            page_content="child text",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        )
    ]

    store.store_child_chunks(
        content_hash="hash-1",
        child_chunks=children,
        embeddings=_HybridEmbeddings(),
    )

    assert sparse_embeddings.calls == [("child text",)]
    assert client.upserts[0]["points"][0]["vector"]["sparse"] == {
        "indices": [10, 20],
        "values": [1.0, 0.5],
    }


def test_store_child_chunks_raises_when_sparse_embeddings_are_missing():
    client = _FakeQdrantClient()
    store = storage.QdrantHybridChildStore(client=client, collection_name="child_chunks_hybrid")
    children = [
        Document(
            page_content="child text",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        )
    ]

    with pytest.raises(ValueError, match="sparse_embeddings"):
        store.store_child_chunks(
            content_hash="hash-1",
            child_chunks=children,
            embeddings=_HybridEmbeddings(),
        )


def test_store_child_chunks_hybrid_creates_collection_when_missing():
    client = _FakeQdrantClient()
    store = storage.QdrantHybridChildStore(client=client, collection_name="child_chunks_hybrid")
    dense_embeddings = _HybridEmbeddings()
    sparse_embeddings = _FakeSparseEncoder()
    children = [
        Document(
            page_content="child text",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        )
    ]

    store.store_child_chunks(
        content_hash="hash-1",
        child_chunks=children,
        embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
    )

    assert len(client.created_collections) == 1
    created = client.created_collections[0]
    assert created["collection_name"] == "child_chunks_hybrid"
    assert created["vectors_config"] is not None
    assert created["sparse_vectors_config"] is not None


def test_store_child_chunks_hybrid_raises_when_dense_vector_count_mismatches_child_chunks():
    client = _FakeQdrantClient()
    store = storage.QdrantHybridChildStore(client=client, collection_name="child_chunks_hybrid")
    children = [
        Document(
            page_content="child text 1",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        ),
        Document(
            page_content="child text 2",
            metadata={
                "child_id": "parent-1-child-1",
                "child_index": 1,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        ),
    ]

    with pytest.raises(ValueError, match="dense vectors"):
        store.store_child_chunks(
            content_hash="hash-1",
            child_chunks=children,
            embeddings=_DenseCountMismatchEmbeddings(),
            sparse_embeddings=_FakeSparseEncoder(),
        )


def test_store_child_chunks_hybrid_raises_when_sparse_vector_count_mismatches_child_chunks():
    client = _FakeQdrantClient()
    store = storage.QdrantHybridChildStore(client=client, collection_name="child_chunks_hybrid")
    children = [
        Document(
            page_content="child text 1",
            metadata={
                "child_id": "parent-1-child-0",
                "child_index": 0,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        ),
        Document(
            page_content="child text 2",
            metadata={
                "child_id": "parent-1-child-1",
                "child_index": 1,
                "parent_id": "parent-1",
                "parent_index": 0,
                "file_path": "demo.txt",
                "source": "demo.txt",
                "file_type": ".txt",
            },
        ),
    ]

    with pytest.raises(ValueError, match="sparse vectors"):
        store.store_child_chunks(
            content_hash="hash-1",
            child_chunks=children,
            embeddings=_HybridEmbeddings(),
            sparse_embeddings=_SparseCountMismatchEncoder(),
        )


def test_mark_failed_updates_error_message():
    client = _FakeMongoClient()
    client["splitter"]["ingested_files"].documents.append(
        {"content_hash": "hash-1", "status": "processing"}
    )
    repo = storage.MongoIngestionRepository(client=client, database_name="splitter")

    repo.mark_failed(content_hash="hash-1", error="qdrant write failed")

    record = client["splitter"]["ingested_files"].documents[0]
    assert record["status"] == "failed"
    assert record["error"] == "qdrant write failed"


def test_build_storage_backend_uses_local_mongo_and_qdrant_defaults(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeMongoClientCtor:
        def __init__(self, uri: str):
            captured["mongo_uri"] = uri
            self.databases: dict[str, _FakeDatabase] = {}

        def __getitem__(self, name: str) -> _FakeDatabase:
            if name not in self.databases:
                self.databases[name] = _FakeDatabase()
            return self.databases[name]

    class _FakeQdrantClientCtor:
        def __init__(self, url: str, check_compatibility: bool, trust_env: bool):
            captured["qdrant_url"] = url
            captured["qdrant_check_compatibility"] = check_compatibility
            captured["qdrant_trust_env"] = trust_env

    class _FakeAsyncMongoClientCtor(_FakeMongoClientCtor):
        def __init__(self, uri: str):
            super().__init__(uri)
            captured["async_mongo_uri"] = uri

    class _FakeAsyncQdrantClientCtor(_FakeQdrantClientCtor):
        def __init__(self, url: str, check_compatibility: bool, trust_env: bool):
            super().__init__(url, check_compatibility, trust_env)
            captured["async_qdrant_url"] = url

    monkeypatch.setattr(storage, "MongoClient", _FakeMongoClientCtor)
    monkeypatch.setattr(storage, "AsyncMongoClient", _FakeAsyncMongoClientCtor)
    monkeypatch.setattr(storage, "QdrantClient", _FakeQdrantClientCtor)
    monkeypatch.setattr(storage, "AsyncQdrantClient", _FakeAsyncQdrantClientCtor)

    backend = storage.build_storage_backend(sparse_embeddings=object())

    assert captured["mongo_uri"] == "mongodb://admin:123456@localhost:27017"
    assert captured["async_mongo_uri"] == "mongodb://admin:123456@localhost:27017"
    assert captured["qdrant_url"] == "http://localhost:6333"
    assert captured["async_qdrant_url"] == "http://localhost:6333"
    assert captured["qdrant_check_compatibility"] is False
    assert captured["qdrant_trust_env"] is False
    assert backend.mongo_repository.database_name == "splitter"
    assert backend.mongo_repository.async_client is not None
    assert backend.qdrant_store.collection_name == "child_chunks_hybrid"
    assert backend.qdrant_store.async_client is not None
