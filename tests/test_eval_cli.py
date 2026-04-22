from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.cli import (
    DEFAULT_EXPERIMENT_NAMES,
    build_experiment_matrix,
    build_parser,
    main,
    parse_args,
    resolve_experiments,
)
from evals.models import EvalSample, RetrievalCheckpoint, StageTrace


class _FakeCollection:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = list(records)
        self.find_calls: list[dict[str, object]] = []

    def find(self, query: dict[str, object]) -> list[dict[str, object]]:
        self.find_calls.append(query)
        return list(self.records)


class _FakeTestset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = list(rows)

    def to_list(self) -> list[dict[str, object]]:
        return list(self.rows)


def _trace_for(sample: EvalSample, experiment_name: str) -> StageTrace:
    return StageTrace(
        sample=sample,
        rewritten_queries=[sample.question],
        retrieval_checkpoints=[
            RetrievalCheckpoint(
                stage_name="collapsed_parents",
                child_ids=[],
                parent_ids=["parent-1"],
                contexts=[f"context for {sample.sample_id}"],
            )
        ],
        final_answer=f"{experiment_name}:{sample.sample_id}:answer",
    )


def test_build_experiment_matrix_includes_required_stage_ablations():
    experiments = build_experiment_matrix()

    assert [config.experiment_name for config in experiments] == list(DEFAULT_EXPERIMENT_NAMES)
    assert experiments[0].to_dict() == {
        "experiment_name": "baseline",
        "enable_query_rewrite": True,
        "enable_multi_query_merge": True,
        "enable_rerank": True,
        "top_k_per_query": 10,
        "final_parent_limit": 5,
    }
    assert experiments[1].experiment_name == "no-rewrite"
    assert experiments[1].enable_query_rewrite is False
    assert experiments[1].enable_multi_query_merge is True
    assert experiments[1].enable_rerank is True
    assert experiments[2].experiment_name == "no-rerank"
    assert experiments[2].enable_query_rewrite is True
    assert experiments[2].enable_rerank is False
    assert experiments[3].experiment_name == "no-multi-query"
    assert experiments[3].enable_query_rewrite is True
    assert experiments[3].enable_multi_query_merge is False


def test_resolve_experiments_returns_named_subset_in_requested_order():
    experiments = resolve_experiments(["no-rerank", "baseline"])

    assert [config.experiment_name for config in experiments] == ["no-rerank", "baseline"]


def test_resolve_experiments_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown experiment"):
        resolve_experiments(["unknown"])


def test_build_parser_accepts_minimal_eval_cli_arguments():
    parser = build_parser()

    args = parser.parse_args(["--dataset", "data/eval.jsonl", "--experiment", "no-rerank"])

    assert args.dataset == "data/eval.jsonl"
    assert args.output_dir == "artifacts/evals"
    assert args.test_size == 10
    assert args.experiments == ["no-rerank"]
    assert args.list_experiments is False


def test_parse_args_defaults_to_all_experiments_when_not_filtered():
    args = parse_args(["--dataset", "data/eval.jsonl"])

    assert args.dataset == "data/eval.jsonl"
    assert args.output_dir == "artifacts/evals"
    assert args.test_size == 10
    assert args.experiments is None
    assert args.list_experiments is False


def test_main_orchestrates_eval_pipeline_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    dataset_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "artifacts"
    parent_chunk_records = [
        {"parent_id": "parent-1", "text": "first parent", "metadata": {"source": "a.md"}},
        {"parent_id": "parent-2", "text": "second parent", "metadata": {"source": "b.md"}},
    ]
    collection = _FakeCollection(parent_chunk_records)
    runtime = SimpleNamespace(
        llm="llm",
        dense_embeddings="dense",
        eval_llm="eval-llm",
        eval_embeddings="eval-embeddings",
        sparse_embeddings="sparse",
        reranker="reranker",
        storage_backend=SimpleNamespace(
            mongo_repository=SimpleNamespace(_parent_chunks=collection),
            qdrant_store=SimpleNamespace(client="qdrant-client", collection_name="child_chunks"),
        ),
    )
    generated_rows = [
        {
            "user_input": "question one",
            "reference": "answer one",
            "reference_contexts": ["context one"],
            "synthesizer_name": "single_hop",
            "difficulty": "easy",
        },
        {
            "user_input": "question two",
            "reference": "answer two",
            "reference_contexts": ["context two"],
            "synthesizer_name": "single_hop",
            "difficulty": "medium",
        },
    ]
    generator_calls: list[dict[str, object]] = []
    trace_calls: list[dict[str, object]] = []
    retrieval_calls: list[dict[str, object]] = []
    generation_calls: list[dict[str, object]] = []
    artifact_calls: list[dict[str, object]] = []

    monkeypatch.setattr("evals.cli.build_eval_runtime", lambda: runtime)
    monkeypatch.setattr(
        "evals.cli.build_ragas_generator",
        lambda *, llm, embedding_model: generator_calls.append(
            {"llm": llm, "embedding_model": embedding_model}
        )
        or "generator",
    )
    monkeypatch.setattr(
        "evals.cli.generate_synthetic_testset",
        lambda records, *, testset_size, generator: generator_calls.append(
            {
                "records": list(records),
                "testset_size": testset_size,
                "generator": generator,
            }
        )
        or _FakeTestset(generated_rows),
    )

    def _fake_run_stage_trace(**kwargs):
        trace_calls.append(kwargs)
        return _trace_for(kwargs["sample"], kwargs["config"].experiment_name)

    monkeypatch.setattr("evals.cli.run_stage_trace", _fake_run_stage_trace)
    monkeypatch.setattr(
        "evals.cli.compute_ragas_retrieval_metrics",
        lambda traces, *, llm: retrieval_calls.append({"traces": list(traces), "llm": llm})
        or {"context_precision": float(len(traces))},
    )
    monkeypatch.setattr(
        "evals.cli.compute_ragas_generation_metrics",
        lambda traces, *, llm, embeddings: generation_calls.append(
            {"traces": list(traces), "llm": llm, "embeddings": embeddings}
        )
        or {"faithfulness": float(len(traces)) / 10.0},
    )

    def _fake_write_experiment_artifacts(output_dir, experiment_name, traces, metrics):
        artifact_calls.append(
            {
                "output_dir": output_dir,
                "experiment_name": experiment_name,
                "traces": list(traces),
                "metrics": dict(metrics),
            }
        )
        experiment_dir = output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        summary_path = experiment_dir / "summary.md"
        trace_path = experiment_dir / "trace.jsonl"
        metrics_path = experiment_dir / "metrics.json"
        summary_path.write_text(f"# {experiment_name}\n", encoding="utf-8")
        trace_path.write_text("\n".join(trace.sample.sample_id for trace in traces) + "\n", encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return {
            "summary_markdown": summary_path,
            "trace_jsonl": trace_path,
            "metrics_json": metrics_path,
        }

    monkeypatch.setattr("evals.cli.write_experiment_artifacts", _fake_write_experiment_artifacts)

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--test-size",
            "2",
            "--experiment",
            "baseline",
            "--experiment",
            "no-rerank",
        ]
    )

    assert exit_code == 0
    assert collection.find_calls == [{}]
    assert generator_calls == [
        {"llm": "llm", "embedding_model": "dense"},
        {
            "records": parent_chunk_records,
            "testset_size": 2,
            "generator": "generator",
        },
    ]

    written_rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    assert written_rows == [
        {
            "sample_id": "sample-1",
            "question": "question one",
            "reference_answer": "answer one",
            "reference_contexts": ["context one"],
            "metadata": {"synthesizer_name": "single_hop", "difficulty": "easy"},
        },
        {
            "sample_id": "sample-2",
            "question": "question two",
            "reference_answer": "answer two",
            "reference_contexts": ["context two"],
            "metadata": {"synthesizer_name": "single_hop", "difficulty": "medium"},
        },
    ]

    assert [
        (call["config"].experiment_name, call["sample"].sample_id) for call in trace_calls
    ] == [
        ("baseline", "sample-1"),
        ("baseline", "sample-2"),
        ("no-rerank", "sample-1"),
        ("no-rerank", "sample-2"),
    ]
    for call in trace_calls:
        assert call["llm"] == "llm"
        assert call["client"] == "qdrant-client"
        assert call["collection_name"] == "child_chunks"
        assert call["embeddings"] == "dense"
        assert call["sparse_embeddings"] == "sparse"
        assert call["mongo_repository"] is runtime.storage_backend.mongo_repository
        assert call["reranker"] == "reranker"

    assert [
        [trace.sample.sample_id for trace in call["traces"]] for call in retrieval_calls
    ] == [["sample-1", "sample-2"], ["sample-1", "sample-2"]]
    assert all(call["llm"] == "eval-llm" for call in retrieval_calls)
    assert [
        [trace.sample.sample_id for trace in call["traces"]] for call in generation_calls
    ] == [["sample-1", "sample-2"], ["sample-1", "sample-2"]]
    assert all(call["llm"] == "eval-llm" for call in generation_calls)
    assert all(call["embeddings"] == "eval-embeddings" for call in generation_calls)

    assert [call["experiment_name"] for call in artifact_calls] == ["baseline", "no-rerank"]
    assert all(call["output_dir"] == output_dir for call in artifact_calls)
    assert all((output_dir / experiment_name / "summary.md").exists() for experiment_name in ("baseline", "no-rerank"))
    assert all((output_dir / experiment_name / "trace.jsonl").exists() for experiment_name in ("baseline", "no-rerank"))
    assert all((output_dir / experiment_name / "metrics.json").exists() for experiment_name in ("baseline", "no-rerank"))


def test_main_lists_experiments_without_running_eval_flow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        "evals.cli.build_eval_runtime",
        lambda: pytest.fail("build_eval_runtime should not be called"),
    )

    exit_code = main(["--list-experiments"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines() == list(DEFAULT_EXPERIMENT_NAMES)


def test_main_requires_dataset_when_running_evals(capsys: pytest.CaptureFixture[str]):
    exit_code = main([])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "the following arguments are required: --dataset" in captured.err
