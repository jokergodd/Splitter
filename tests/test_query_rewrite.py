from __future__ import annotations

import asyncio
from types import SimpleNamespace

from rag_demo.query_rewrite import QueryRewriteResult, rewrite_queries, rewrite_queries_async


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "\n".join(self.responses)


class _FakeAsyncLLM(_FakeLLM):
    async def ainvoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "\n".join(self.responses)


def test_query_rewrite_result_is_a_simple_dataclass():
    result = QueryRewriteResult(original_query="原始问题", rewritten_queries=["原始问题", "候选问题"])

    assert result.original_query == "原始问题"
    assert result.rewritten_queries == ["原始问题", "候选问题"]


def test_rewrite_queries_keeps_original_first_and_deduplicates_llm_candidates():
    llm = _FakeLLM(["  候选一  ", "", "原始问题", "候选一", "候选二  "])

    result = rewrite_queries("原始问题", llm, max_queries=4)

    assert result.original_query == "原始问题"
    assert result.rewritten_queries == ["原始问题", "候选一", "候选二"]
    assert llm.calls


def test_rewrite_queries_trims_empty_and_limits_total_results():
    llm = _FakeLLM(["  候选一  ", " 候选二 ", "候选三 ", " 候选四 ", " 候选五 "])

    result = rewrite_queries("原始问题", llm, max_queries=3)

    assert result.rewritten_queries == ["原始问题", "候选一", "候选二"]


def test_rewrite_queries_accepts_message_like_llm_response():
    llm = _FakeLLM([])
    llm.invoke = lambda prompt: SimpleNamespace(content="候选一\n候选二")

    result = rewrite_queries("原始问题", llm, max_queries=4)

    assert result.rewritten_queries == ["原始问题", "候选一", "候选二"]


def test_rewrite_queries_async_prefers_ainvoke():
    llm = _FakeAsyncLLM(["候选一", "候选二"])

    result = asyncio.run(rewrite_queries_async("原始问题", llm, max_queries=4))

    assert result.rewritten_queries == ["原始问题", "候选一", "候选二"]
    assert len(llm.calls) == 1
