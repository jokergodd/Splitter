from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from evals.models import EvalSample, ExperimentConfig
from evals.stage_runner import run_stage_trace
from rag_demo.retrieval import HybridRetrievalHit


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="sample-1",
        question="how does stage trace work?",
        reference_answer="reference",
        reference_contexts=["context"],
    )


def _hit(child_id: str, parent_id: str, score: float, text: str) -> HybridRetrievalHit:
    return HybridRetrievalHit(
        child_id=child_id,
        score=score,
        payload={"child_id": child_id, "parent_id": parent_id, "text": text},
        point_id=f"point-{child_id}",
    )


def test_run_stage_trace_records_required_stages(monkeypatch):
    llm = _FakeLLM("final answer")
    per_query_hits = {
        "how does stage trace work?": [
            _hit("child-1", "parent-1", 0.4, "child-one"),
            _hit("child-2", "parent-2", 0.3, "child-two"),
        ],
        "stage trace skeleton": [
            _hit("child-3", "parent-3", 0.9, "child-three"),
            _hit("child-1", "parent-1", 0.5, "child-one-better"),
        ],
    }
    reranked = [
        Document(
            page_content="child-three",
            metadata={"child_id": "child-3", "parent_id": "parent-3", "rerank_score": 0.95},
            id="point-child-3",
        ),
        Document(
            page_content="child-one-better",
            metadata={"child_id": "child-1", "parent_id": "parent-1", "rerank_score": 0.6},
            id="point-child-1",
        ),
    ]
    collapsed = [
        Document(
            page_content="parent-three-hit",
            metadata={"child_id": "child-3", "parent_id": "parent-3", "rerank_score": 0.95},
            id="parent-hit-3",
        ),
        Document(
            page_content="parent-one-hit",
            metadata={"child_id": "child-1", "parent_id": "parent-1", "rerank_score": 0.6},
            id="parent-hit-1",
        ),
    ]
    parent_chunks = [
        Document(page_content="parent three", metadata={"parent_id": "parent-3", "source": "doc-3"}),
        Document(page_content="parent one", metadata={"parent_id": "parent-1", "source": "doc-1"}),
    ]

    monkeypatch.setattr(
        "evals.stage_runner.rewrite_queries",
        lambda query, llm, max_queries=4: SimpleNamespace(
            rewritten_queries=[query, "stage trace skeleton"]
        ),
    )
    monkeypatch.setattr(
        "evals.stage_runner.query_hybrid_children",
        lambda **kwargs: per_query_hits[kwargs["query_text"]],
    )
    monkeypatch.setattr(
        "evals.stage_runner.rerank_candidates",
        lambda original_query, candidates, reranker, limit=10: reranked,
    )
    monkeypatch.setattr(
        "evals.stage_runner.collapse_to_parent_hits",
        lambda reranked_candidates, limit=5: collapsed,
    )
    monkeypatch.setattr(
        "evals.stage_runner.fetch_parent_chunks",
        lambda parent_ids, mongo_repository: parent_chunks,
    )
    monkeypatch.setattr(
        "evals.stage_runner.build_answer_prompt",
        lambda original_query, parent_chunks: f"PROMPT::{original_query}::{len(parent_chunks)}",
    )

    trace = run_stage_trace(
        sample=_sample(),
        config=ExperimentConfig(experiment_name="baseline"),
        llm=llm,
        client=SimpleNamespace(),
        collection_name="child_chunks_hybrid",
        embeddings=SimpleNamespace(),
        sparse_embeddings=SimpleNamespace(),
        mongo_repository=SimpleNamespace(),
        reranker=SimpleNamespace(),
    )

    assert trace.rewritten_queries == ["how does stage trace work?", "stage trace skeleton"]
    assert trace.final_answer == "final answer"
    assert llm.calls == ["PROMPT::how does stage trace work?::2"]

    stage_names = [checkpoint.stage_name for checkpoint in trace.retrieval_checkpoints]
    assert stage_names == [
        "hybrid_per_query",
        "hybrid_per_query",
        "merged_candidates",
        "reranked_candidates",
        "collapsed_parents",
    ]

    first_query_checkpoint = trace.retrieval_checkpoints[0]
    assert first_query_checkpoint.query_text == "how does stage trace work?"
    assert first_query_checkpoint.child_ids == ["child-1", "child-2"]
    assert first_query_checkpoint.parent_ids == ["parent-1", "parent-2"]
    assert first_query_checkpoint.items == [
        {
            "rank": 1,
            "child_id": "child-1",
            "parent_id": "parent-1",
            "score": 0.4,
            "point_id": "point-child-1",
            "text": "child-one",
        },
        {
            "rank": 2,
            "child_id": "child-2",
            "parent_id": "parent-2",
            "score": 0.3,
            "point_id": "point-child-2",
            "text": "child-two",
        },
    ]

    merged_checkpoint = trace.retrieval_checkpoints[2]
    assert merged_checkpoint.child_ids == ["child-3", "child-1", "child-2"]
    assert merged_checkpoint.items == [
        {
            "rank": 1,
            "child_id": "child-3",
            "parent_id": "parent-3",
            "score": 0.9,
            "point_id": "point-child-3",
            "text": "child-three",
            "provenance": [
                {"query_text": "stage trace skeleton", "rank": 1, "score": 0.9},
            ],
        },
        {
            "rank": 2,
            "child_id": "child-1",
            "parent_id": "parent-1",
            "score": 0.5,
            "point_id": "point-child-1",
            "text": "child-one-better",
            "provenance": [
                {"query_text": "how does stage trace work?", "rank": 1, "score": 0.4},
                {"query_text": "stage trace skeleton", "rank": 2, "score": 0.5},
            ],
        },
        {
            "rank": 3,
            "child_id": "child-2",
            "parent_id": "parent-2",
            "score": 0.3,
            "point_id": "point-child-2",
            "text": "child-two",
            "provenance": [
                {"query_text": "how does stage trace work?", "rank": 2, "score": 0.3},
            ],
        },
    ]

    reranked_checkpoint = trace.retrieval_checkpoints[3]
    assert reranked_checkpoint.child_ids == ["child-3", "child-1"]
    assert reranked_checkpoint.parent_ids == ["parent-3", "parent-1"]
    assert reranked_checkpoint.items == [
        {
            "rank": 1,
            "child_id": "child-3",
            "parent_id": "parent-3",
            "rerank_score": 0.95,
            "retrieval_score": None,
            "point_id": "point-child-3",
            "text": "child-three",
        },
        {
            "rank": 2,
            "child_id": "child-1",
            "parent_id": "parent-1",
            "rerank_score": 0.6,
            "retrieval_score": None,
            "point_id": "point-child-1",
            "text": "child-one-better",
        },
    ]

    collapsed_checkpoint = trace.retrieval_checkpoints[4]
    assert collapsed_checkpoint.parent_ids == ["parent-3", "parent-1"]
    assert collapsed_checkpoint.contexts == ["parent three", "parent one"]
    assert collapsed_checkpoint.items == [
        {
            "rank": 1,
            "parent_id": "parent-3",
            "child_id": "child-3",
            "rerank_score": 0.95,
            "point_id": "parent-hit-3",
            "child_text": "parent-three-hit",
            "parent_found": True,
            "parent_text": "parent three",
        },
        {
            "rank": 2,
            "parent_id": "parent-1",
            "child_id": "child-1",
            "rerank_score": 0.6,
            "point_id": "parent-hit-1",
            "child_text": "parent-one-hit",
            "parent_found": True,
            "parent_text": "parent one",
        },
    ]


def test_run_stage_trace_respects_config_toggles(monkeypatch):
    llm = _FakeLLM("no-rerank answer")
    rewrite_called = False
    rerank_called = False

    def _rewrite(*args, **kwargs):
        nonlocal rewrite_called
        rewrite_called = True
        return SimpleNamespace(rewritten_queries=["unexpected"])

    def _rerank(*args, **kwargs):
        nonlocal rerank_called
        rerank_called = True
        return []

    monkeypatch.setattr("evals.stage_runner.rewrite_queries", _rewrite)
    monkeypatch.setattr(
        "evals.stage_runner.query_hybrid_children",
        lambda **kwargs: [_hit("child-1", "parent-1", 0.4, "child-one")],
    )
    monkeypatch.setattr("evals.stage_runner.rerank_candidates", _rerank)
    monkeypatch.setattr(
        "evals.stage_runner.collapse_to_parent_hits",
        lambda reranked_candidates, limit=5: [
            Document(
                page_content=reranked_candidates[0].page_content,
                metadata=dict(reranked_candidates[0].metadata),
                id=reranked_candidates[0].id,
            )
        ],
    )
    monkeypatch.setattr(
        "evals.stage_runner.fetch_parent_chunks",
        lambda parent_ids, mongo_repository: [
            Document(page_content="parent one", metadata={"parent_id": "parent-1"})
        ],
    )
    monkeypatch.setattr(
        "evals.stage_runner.build_answer_prompt",
        lambda original_query, parent_chunks: f"PROMPT::{original_query}",
    )

    trace = run_stage_trace(
        sample=_sample(),
        config=ExperimentConfig(
            experiment_name="ablated",
            enable_query_rewrite=False,
            enable_multi_query_merge=False,
            enable_rerank=False,
            top_k_per_query=3,
            final_parent_limit=1,
        ),
        llm=llm,
        client=SimpleNamespace(),
        collection_name="child_chunks_hybrid",
        embeddings=SimpleNamespace(),
        sparse_embeddings=SimpleNamespace(),
        mongo_repository=SimpleNamespace(),
        reranker=SimpleNamespace(),
    )

    assert rewrite_called is False
    assert rerank_called is False
    assert trace.rewritten_queries == ["how does stage trace work?"]
    assert [checkpoint.stage_name for checkpoint in trace.retrieval_checkpoints] == [
        "hybrid_per_query",
        "merged_candidates",
        "reranked_candidates",
        "collapsed_parents",
    ]
    assert trace.retrieval_checkpoints[0].query_text == "how does stage trace work?"
    assert trace.retrieval_checkpoints[0].items == [
        {
            "rank": 1,
            "child_id": "child-1",
            "parent_id": "parent-1",
            "score": 0.4,
            "point_id": "point-child-1",
            "text": "child-one",
        }
    ]
    assert trace.retrieval_checkpoints[1].child_ids == ["child-1"]
    assert trace.retrieval_checkpoints[2].child_ids == ["child-1"]
    assert trace.retrieval_checkpoints[3].contexts == ["parent one"]
    assert trace.retrieval_checkpoints[3].items == [
        {
            "rank": 1,
            "parent_id": "parent-1",
            "child_id": "child-1",
            "rerank_score": None,
            "point_id": "point-child-1",
            "child_text": "child-one",
            "parent_found": True,
            "parent_text": "parent one",
        }
    ]


def test_run_stage_trace_accepts_llm_message_objects(monkeypatch):
    llm = SimpleNamespace(invoke=lambda prompt: _FakeMessage("message answer"))

    monkeypatch.setattr(
        "evals.stage_runner.query_hybrid_children",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "evals.stage_runner.fetch_parent_chunks",
        lambda parent_ids, mongo_repository: [],
    )
    monkeypatch.setattr(
        "evals.stage_runner.build_answer_prompt",
        lambda original_query, parent_chunks: "PROMPT",
    )

    trace = run_stage_trace(
        sample=_sample(),
        config=ExperimentConfig(
            experiment_name="message",
            enable_query_rewrite=False,
            enable_rerank=False,
        ),
        llm=llm,
        client=SimpleNamespace(),
        collection_name="child_chunks_hybrid",
        embeddings=SimpleNamespace(),
        sparse_embeddings=SimpleNamespace(),
        mongo_repository=SimpleNamespace(),
    )

    assert trace.final_answer == "message answer"


def test_run_stage_trace_keeps_collapse_evidence_when_parent_fetch_is_missing(monkeypatch):
    llm = _FakeLLM("partial fetch answer")

    monkeypatch.setattr(
        "evals.stage_runner.query_hybrid_children",
        lambda **kwargs: [
            _hit("child-1", "parent-1", 0.4, "child-one"),
            _hit("child-2", "parent-2", 0.3, "child-two"),
        ],
    )
    monkeypatch.setattr(
        "evals.stage_runner.rerank_candidates",
        lambda original_query, candidates, reranker, limit=10: [
            Document(
                page_content="child-one",
                metadata={"child_id": "child-1", "parent_id": "parent-1", "rerank_score": 0.9},
                id="point-child-1",
            ),
            Document(
                page_content="child-two",
                metadata={"child_id": "child-2", "parent_id": "parent-2", "rerank_score": 0.8},
                id="point-child-2",
            ),
        ],
    )
    monkeypatch.setattr(
        "evals.stage_runner.collapse_to_parent_hits",
        lambda reranked_candidates, limit=5: [
            Document(
                page_content="collapsed-one",
                metadata={"child_id": "child-1", "parent_id": "parent-1", "rerank_score": 0.9},
                id="parent-hit-1",
            ),
            Document(
                page_content="collapsed-two",
                metadata={"child_id": "child-2", "parent_id": "parent-2", "rerank_score": 0.8},
                id="parent-hit-2",
            ),
        ],
    )
    monkeypatch.setattr(
        "evals.stage_runner.fetch_parent_chunks",
        lambda parent_ids, mongo_repository: [
            Document(page_content="parent one", metadata={"parent_id": "parent-1"})
        ],
    )
    monkeypatch.setattr(
        "evals.stage_runner.build_answer_prompt",
        lambda original_query, parent_chunks: "PROMPT",
    )

    trace = run_stage_trace(
        sample=_sample(),
        config=ExperimentConfig(
            experiment_name="partial-parent-fetch",
            enable_query_rewrite=False,
        ),
        llm=llm,
        client=SimpleNamespace(),
        collection_name="child_chunks_hybrid",
        embeddings=SimpleNamespace(),
        sparse_embeddings=SimpleNamespace(),
        mongo_repository=SimpleNamespace(),
        reranker=SimpleNamespace(),
    )

    collapsed_checkpoint = trace.retrieval_checkpoints[-1]
    assert collapsed_checkpoint.parent_ids == ["parent-1", "parent-2"]
    assert collapsed_checkpoint.contexts == ["parent one", ""]
    assert collapsed_checkpoint.items == [
        {
            "rank": 1,
            "parent_id": "parent-1",
            "child_id": "child-1",
            "rerank_score": 0.9,
            "point_id": "parent-hit-1",
            "child_text": "collapsed-one",
            "parent_found": True,
            "parent_text": "parent one",
        },
        {
            "rank": 2,
            "parent_id": "parent-2",
            "child_id": "child-2",
            "rerank_score": 0.8,
            "point_id": "parent-hit-2",
            "child_text": "collapsed-two",
            "parent_found": False,
            "parent_text": None,
        },
    ]
