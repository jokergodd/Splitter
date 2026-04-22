from __future__ import annotations

import json
from pathlib import Path

from evals.models import EvalSample, RetrievalCheckpoint, StageTrace
from evals.reporting import (
    write_experiment_artifacts,
    write_metrics_json,
    write_summary_markdown,
    write_trace_jsonl,
)


def _trace(sample_id: str = "sample-1") -> StageTrace:
    return StageTrace(
        sample=EvalSample(
            sample_id=sample_id,
            question="Where are parent chunks stored?",
            reference_answer="Parent chunks are stored in MongoDB.",
            reference_contexts=["Parent chunks are stored in MongoDB."],
            metadata={"source": "storage.md"},
        ),
        rewritten_queries=["Where are parent chunks stored?"],
        retrieval_checkpoints=[
            RetrievalCheckpoint(
                stage_name="collapsed_parents",
                child_ids=["child-1"],
                parent_ids=["parent-1"],
                contexts=["Parent chunks are stored in MongoDB."],
                query_text="Where are parent chunks stored?",
                items=[{"parent_id": "parent-1", "score": 0.98}],
            )
        ],
        final_answer="Parent chunks are stored in MongoDB.",
    )


def test_write_summary_markdown_outputs_metric_sections(tmp_path: Path):
    output_path = tmp_path / "summary.md"

    write_summary_markdown(
        output_path=output_path,
        experiment_name="baseline",
        metrics={
            "retrieval": {"context_recall": 0.75, "context_precision": 0.5},
            "generation": {"faithfulness": 0.81},
        },
        trace_count=2,
    )

    content = output_path.read_text(encoding="utf-8")

    assert "# baseline" in content
    assert "Trace count: 2" in content
    assert "## retrieval" in content
    assert "| context_recall | 0.7500 |" in content
    assert "## generation" in content
    assert "| faithfulness | 0.8100 |" in content


def test_write_trace_jsonl_serializes_each_trace_on_its_own_line(tmp_path: Path):
    output_path = tmp_path / "trace.jsonl"

    write_trace_jsonl(output_path=output_path, traces=[_trace("sample-1"), _trace("sample-2")])

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    first_payload = json.loads(lines[0])
    second_payload = json.loads(lines[1])
    assert first_payload["sample"]["sample_id"] == "sample-1"
    assert second_payload["sample"]["sample_id"] == "sample-2"
    assert first_payload["retrieval_checkpoints"][0]["stage_name"] == "collapsed_parents"


def test_write_metrics_json_preserves_nested_metrics_payload(tmp_path: Path):
    output_path = tmp_path / "metrics.json"
    metrics = {
        "retrieval": {"context_recall": 0.75},
        "generation": {"faithfulness": 0.81},
    }

    write_metrics_json(output_path=output_path, metrics=metrics)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload == metrics


def test_write_experiment_artifacts_writes_expected_files(tmp_path: Path):
    output_dir = tmp_path / "artifacts"
    trace = _trace()
    metrics = {"retrieval": {"context_recall": 0.75}, "generation": {"faithfulness": 0.81}}

    paths = write_experiment_artifacts(
        output_dir=output_dir,
        experiment_name="baseline",
        traces=[trace],
        metrics=metrics,
    )

    assert paths == {
        "summary_markdown": output_dir / "baseline" / "summary.md",
        "trace_jsonl": output_dir / "baseline" / "trace.jsonl",
        "metrics_json": output_dir / "baseline" / "metrics.json",
    }
    assert paths["summary_markdown"].exists()
    assert paths["trace_jsonl"].exists()
    assert paths["metrics_json"].exists()
