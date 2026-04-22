from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _to_serializable_trace(trace: Any) -> dict[str, Any]:
    if hasattr(trace, "to_dict"):
        return trace.to_dict()
    if isinstance(trace, dict):
        return trace
    raise TypeError("trace must be a dict or expose a to_dict() method")


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary_markdown(
    output_path: Path,
    experiment_name: str,
    metrics: Mapping[str, Mapping[str, Any]],
    trace_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# {experiment_name}", "", f"Trace count: {trace_count}", ""]
    for section_name, section_metrics in metrics.items():
        lines.extend([f"## {section_name}", "", "| metric | value |", "| --- | --- |"])
        for metric_name, metric_value in section_metrics.items():
            lines.append(f"| {metric_name} | {_format_metric_value(metric_value)} |")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_trace_jsonl(output_path: Path, traces: Sequence[Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for trace in traces:
            payload = _to_serializable_trace(trace)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_metrics_json(output_path: Path, metrics: Mapping[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(metrics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_experiment_artifacts(
    output_dir: Path,
    experiment_name: str,
    traces: Sequence[Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    experiment_dir = output_dir / experiment_name
    paths = {
        "summary_markdown": experiment_dir / "summary.md",
        "trace_jsonl": experiment_dir / "trace.jsonl",
        "metrics_json": experiment_dir / "metrics.json",
    }

    write_summary_markdown(
        output_path=paths["summary_markdown"],
        experiment_name=experiment_name,
        metrics=metrics,
        trace_count=len(traces),
    )
    write_trace_jsonl(output_path=paths["trace_jsonl"], traces=traces)
    write_metrics_json(output_path=paths["metrics_json"], metrics=metrics)

    return paths


__all__ = [
    "write_experiment_artifacts",
    "write_metrics_json",
    "write_summary_markdown",
    "write_trace_jsonl",
]
