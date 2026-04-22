from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryRewriteResult:
    original_query: str
    rewritten_queries: list[str] = field(default_factory=list)


def _rewrite_prompt(query: str) -> str:
    return (
        "Rewrite the following query into short alternative search queries, "
        "one per line, without numbering or extra commentary:\n"
        f"{query}"
    )


def _rewrite_result(query: str, raw_output: Any, *, max_queries: int) -> QueryRewriteResult:
    limit = max(1, max_queries)

    if isinstance(raw_output, str):
        candidates = raw_output.splitlines()
    elif hasattr(raw_output, "content") and isinstance(raw_output.content, str):
        candidates = raw_output.content.splitlines()
    elif raw_output is None:
        candidates = []
    else:
        candidates = list(raw_output)

    rewritten_queries: list[str] = [query]
    seen = {query}

    for candidate in candidates:
        normalized = str(candidate).strip()
        if not normalized or normalized in seen:
            continue
        rewritten_queries.append(normalized)
        seen.add(normalized)
        if len(rewritten_queries) >= limit:
            break

    return QueryRewriteResult(original_query=query, rewritten_queries=rewritten_queries[:limit])


def rewrite_queries(query: str, llm: Any, max_queries: int = 4) -> QueryRewriteResult:
    raw_output = llm.invoke(_rewrite_prompt(query))
    return _rewrite_result(query, raw_output, max_queries=max_queries)


async def rewrite_queries_async(query: str, llm: Any, max_queries: int = 4) -> QueryRewriteResult:
    prompt = _rewrite_prompt(query)
    if hasattr(llm, "ainvoke"):
        raw_output = await llm.ainvoke(prompt)
    else:
        raw_output = await asyncio.to_thread(llm.invoke, prompt)
    return _rewrite_result(query, raw_output, max_queries=max_queries)
