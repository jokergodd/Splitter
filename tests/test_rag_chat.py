from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import rag_chat


def test_build_runtime_initializes_all_runtime_dependencies_once(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDenseEmbeddings:
        def __init__(self, *, model_name: str):
            captured["dense_model_name"] = model_name

    class FakeCachedEmbeddings:
        def __init__(self, base_embeddings):
            captured["dense_base_embeddings"] = base_embeddings
            self.base_embeddings = base_embeddings

    class FakeSparseEmbeddings:
        def __init__(self, *, model_name: str):
            captured["sparse_model_name"] = model_name

    fake_llm = object()
    fake_reranker = object()
    fake_storage_backend = object()

    monkeypatch.setattr(rag_chat, "load_deepseek_config", lambda: "config")
    monkeypatch.setattr(rag_chat, "build_deepseek_llm", lambda config: fake_llm)
    monkeypatch.setattr(rag_chat, "HuggingFaceEmbeddings", FakeDenseEmbeddings)
    monkeypatch.setattr(rag_chat, "CachedEmbeddings", FakeCachedEmbeddings)
    monkeypatch.setattr(rag_chat, "SparseTextEmbedding", FakeSparseEmbeddings)
    monkeypatch.setattr(rag_chat, "build_cross_encoder_reranker", lambda: fake_reranker)
    monkeypatch.setattr(
        rag_chat,
        "build_storage_backend",
        lambda *, sparse_embeddings: captured.__setitem__("storage_sparse_embeddings", sparse_embeddings)
        or fake_storage_backend,
    )

    runtime = rag_chat.build_runtime()

    assert runtime.llm is fake_llm
    assert isinstance(runtime.dense_embeddings, FakeCachedEmbeddings)
    assert runtime.dense_embeddings.base_embeddings.__class__ is FakeDenseEmbeddings
    assert runtime.eval_llm is None
    assert runtime.eval_embeddings is None
    assert isinstance(runtime.sparse_embeddings, FakeSparseEmbeddings)
    assert runtime.reranker is fake_reranker
    assert runtime.storage_backend is fake_storage_backend
    assert captured == {
        "dense_model_name": rag_chat.DEFAULT_DENSE_MODEL_NAME,
        "dense_base_embeddings": runtime.dense_embeddings.base_embeddings,
        "sparse_model_name": rag_chat.DEFAULT_SPARSE_MODEL_NAME,
        "storage_sparse_embeddings": runtime.sparse_embeddings,
    }


def test_main_runs_repl_and_prints_answer_and_sources_then_exits(monkeypatch, capsys):
    runtime = SimpleNamespace(llm=object(), dense_embeddings=object(), sparse_embeddings=object(), reranker=object(), storage_backend=object())
    result = SimpleNamespace(
        answer="final answer",
        parent_chunks=[Document(page_content="parent text", metadata={"parent_id": "parent-1"})],
        source_items=[
            {
                "parent_id": "parent-1",
                "source": "demo.pdf",
                "file_path": "C:/docs/demo.pdf",
            }
        ],
    )
    captured_runtime: list[object] = []
    calls: list[str] = []

    class FakeChatService:
        def __init__(self, runtime_arg):
            captured_runtime.append(runtime_arg)

        async def answer(self, *, question: str):
            calls.append(question)
            return result

    monkeypatch.setattr(rag_chat, "build_runtime", lambda: runtime)
    monkeypatch.setattr(rag_chat, "ChatService", FakeChatService)

    prompts: list[str] = []
    inputs = iter(["hello", "quit"])

    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(inputs))

    exit_code = rag_chat.main()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured_runtime == [runtime]
    assert calls == ["hello"]
    assert prompts == ["Question> ", "Question> "]
    assert "Answer:" in output
    assert "final answer" in output
    assert "Sources:" in output
    assert "parent_id=parent-1" in output
    assert "source=demo.pdf" in output
    assert "file_path=C:/docs/demo.pdf" in output


def test_main_continues_after_answering_error(monkeypatch, capsys):
    runtime = SimpleNamespace(llm=object(), dense_embeddings=object(), sparse_embeddings=object(), reranker=object(), storage_backend=object())
    result = SimpleNamespace(
        answer="recovered answer",
        parent_chunks=[Document(page_content="parent text", metadata={"parent_id": "parent-2"})],
        source_items=[
            {
                "parent_id": "parent-2",
                "source": "demo.txt",
                "file_path": "C:/docs/demo.txt",
            }
        ],
    )
    calls: list[str] = []

    class FakeChatService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, *, question: str):
            calls.append(question)
            if question == "bad":
                raise RuntimeError("boom")
            return result

    monkeypatch.setattr(rag_chat, "build_runtime", lambda: runtime)
    monkeypatch.setattr(rag_chat, "ChatService", FakeChatService)

    inputs = iter(["bad", "good", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    exit_code = rag_chat.main()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == ["bad", "good"]
    assert "Error: boom" in output
    assert "recovered answer" in output
    assert "parent_id=parent-2" in output


def test_main_uses_asyncio_run_with_chat_service(monkeypatch):
    runtime = SimpleNamespace(llm=object(), dense_embeddings=object(), sparse_embeddings=object(), reranker=object(), storage_backend=object())
    awaited: list[str] = []
    seen_coroutines: list[object] = []
    original_asyncio_run = asyncio.run

    class FakeChatService:
        def __init__(self, runtime_arg):
            assert runtime_arg is runtime

        async def answer(self, *, question: str):
            awaited.append(question)
            return SimpleNamespace(answer="ok", source_items=[])

    def fake_asyncio_run(coro):
        seen_coroutines.append(coro)
        return original_asyncio_run(coro)

    monkeypatch.setattr(rag_chat, "build_runtime", lambda: runtime)
    monkeypatch.setattr(rag_chat, "ChatService", FakeChatService)
    monkeypatch.setattr(rag_chat.asyncio, "run", fake_asyncio_run)
    inputs = iter(["hello", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    exit_code = rag_chat.main()

    assert exit_code == 0
    assert awaited == ["hello"]
    assert len(seen_coroutines) == 1
