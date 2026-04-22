from __future__ import annotations

from pathlib import Path

import pytest

from evals.models import EvalSample, ExperimentConfig, RetrievalCheckpoint, StageTrace


def test_experiment_config_exposes_core_ablation_fields():
    config = ExperimentConfig(
        experiment_name="camera-rag-baseline",
        enable_query_rewrite=False,
        enable_multi_query_merge=True,
        enable_rerank=False,
        top_k_per_query=7,
        final_parent_limit=3,
    )

    payload = config.to_dict()

    assert payload["experiment_name"] == "camera-rag-baseline"
    assert payload["enable_query_rewrite"] is False
    assert payload["enable_multi_query_merge"] is True
    assert payload["enable_rerank"] is False
    assert payload["top_k_per_query"] == 7
    assert payload["final_parent_limit"] == 3


def test_stage_trace_serializes_nested_evaluation_records():
    sample = EvalSample(
        sample_id="sample-1",
        question="光线对人像摄影有什么影响？",
        reference_answer="会影响曝光、阴影、轮廓和情绪表达。",
        reference_contexts=["光线方向会改变面部阴影。"],
        metadata={"file_type": ".pdf"},
    )
    trace = StageTrace(
        sample=sample,
        rewritten_queries=["光线对人像摄影有什么影响？"],
        retrieval_checkpoints=[
            RetrievalCheckpoint(
                stage_name="hybrid",
                child_ids=["child-1"],
                parent_ids=["parent-1"],
                contexts=["光线方向会改变面部阴影。"],
            )
        ],
        final_answer="会影响曝光、阴影、轮廓和情绪表达。",
    )

    payload = trace.to_dict()

    assert payload["sample"]["sample_id"] == "sample-1"
    assert payload["sample"]["metadata"] == {"file_type": ".pdf"}
    assert payload["retrieval_checkpoints"][0]["stage_name"] == "hybrid"
    assert payload["retrieval_checkpoints"][0]["child_ids"] == ["child-1"]
    assert payload["final_answer"] == "会影响曝光、阴影、轮廓和情绪表达。"


def test_stage_trace_rejects_non_json_serializable_values():
    sample = EvalSample(
        sample_id="sample-2",
        question="demo",
        reference_answer="demo",
        reference_contexts=[],
        metadata={"artifact": Path("demo.txt")},
    )
    trace = StageTrace(
        sample=sample,
        rewritten_queries=[],
        retrieval_checkpoints=[],
        final_answer="demo",
    )

    with pytest.raises(TypeError, match="not JSON-serializable"):
        trace.to_dict()
