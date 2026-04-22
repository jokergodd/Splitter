from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

from evals.dataset_generation import build_ragas_generator, generate_synthetic_testset
from evals.metrics_generation import compute_ragas_generation_metrics
from evals.metrics_retrieval import compute_ragas_retrieval_metrics
from evals.models import EvalSample, ExperimentConfig
from evals.reporting import write_experiment_artifacts
from evals.stage_runner import run_stage_trace

DEFAULT_EVAL_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


DEFAULT_EXPERIMENT_NAMES: tuple[str, ...] = (
    "baseline",
    "no-rewrite",
    "no-rerank",
    "no-multi-query",
)


def build_experiment_matrix() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(experiment_name="baseline"),
        ExperimentConfig(experiment_name="no-rewrite", enable_query_rewrite=False),
        ExperimentConfig(experiment_name="no-rerank", enable_rerank=False),
        ExperimentConfig(
            experiment_name="no-multi-query",
            enable_multi_query_merge=False,
        ),
    ]


def resolve_experiments(experiment_names: Sequence[str] | None = None) -> list[ExperimentConfig]:
    experiments_by_name = {
        config.experiment_name: config for config in build_experiment_matrix()
    }
    if experiment_names is None:
        return list(experiments_by_name.values())

    resolved: list[ExperimentConfig] = []
    for experiment_name in experiment_names:
        config = experiments_by_name.get(experiment_name)
        if config is None:
            available = ", ".join(experiments_by_name)
            raise ValueError(
                f"Unknown experiment '{experiment_name}'. Available experiments: {available}"
            )
        resolved.append(config)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure evaluation ablation runs.")
    parser.add_argument("--dataset", help="Path to write the generated evaluation dataset file.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/evals",
        help="Directory where per-experiment evaluation artifacts will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="Number of synthetic evaluation samples to generate from parent chunks.",
    )
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        choices=DEFAULT_EXPERIMENT_NAMES,
        help="Experiment name to run. Repeat to select multiple ablations.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiment names without running orchestration.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_eval_runtime() -> Any:
    from rag_chat import build_runtime
    from rag_demo.llm import build_ragas_eval_llm, load_deepseek_config

    runtime = build_runtime()
    config = load_deepseek_config()
    runtime.eval_llm = build_ragas_eval_llm(config)
    runtime.eval_embeddings = RagasHuggingFaceEmbeddings(
        model=DEFAULT_EVAL_EMBEDDING_MODEL_NAME
    )
    return runtime


def load_parent_chunk_records(mongo_repository: Any) -> list[dict[str, Any]]:
    collection = getattr(mongo_repository, "_parent_chunks", None)
    if collection is None:
        raise AttributeError("mongo_repository must expose _parent_chunks")
    return [dict(record) for record in collection.find({})]


def _coerce_row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "model_dump"):
        return dict(row.model_dump(exclude_none=True))
    if hasattr(row, "dict"):
        return dict(row.dict())
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    raise TypeError("synthetic sample row must be a mapping or expose model_dump()/dict()")


def _sample_rows(testset: Any) -> list[dict[str, Any]]:
    if hasattr(testset, "to_list"):
        rows = testset.to_list()
    elif isinstance(testset, Iterable) and not isinstance(testset, (str, bytes, Mapping)):
        rows = list(testset)
    else:
        rows = [testset]
    return [_coerce_row_mapping(row) for row in rows]


def _normalize_reference_contexts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise TypeError("reference_contexts must be a string, sequence, or None")


def _normalize_eval_sample(row: Mapping[str, Any], *, index: int) -> EvalSample:
    data = dict(row)

    question = data.pop("user_input", data.pop("question", None))
    if question is None:
        raise ValueError("synthetic sample row is missing user_input")

    reference_answer = data.pop("reference", data.pop("reference_answer", ""))
    reference_contexts = _normalize_reference_contexts(
        data.pop("reference_contexts", data.pop("contexts", []))
    )
    sample_id = str(data.pop("sample_id", data.pop("id", f"sample-{index}")))

    return EvalSample(
        sample_id=sample_id,
        question=str(question),
        reference_answer=str(reference_answer),
        reference_contexts=reference_contexts,
        metadata=data,
    )


def build_eval_samples(*, runtime: Any, parent_chunk_records: Sequence[Mapping[str, Any]], test_size: int) -> list[EvalSample]:
    generator = build_ragas_generator(
        llm=runtime.llm,
        embedding_model=runtime.dense_embeddings,
    )
    synthetic_testset = generate_synthetic_testset(
        parent_chunk_records,
        testset_size=test_size,
        generator=generator,
    )
    return [
        _normalize_eval_sample(row, index=index)
        for index, row in enumerate(_sample_rows(synthetic_testset), start=1)
    ]


def write_eval_dataset(output_path: Path, samples: Sequence[EvalSample]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def run_experiment(
    *,
    runtime: Any,
    samples: Sequence[EvalSample],
    experiment: ExperimentConfig,
    output_dir: Path,
) -> dict[str, Any]:
    storage_backend = runtime.storage_backend
    qdrant_store = storage_backend.qdrant_store
    metrics_llm = getattr(runtime, "eval_llm", runtime.llm)
    metrics_embeddings = getattr(runtime, "eval_embeddings", runtime.dense_embeddings)

    traces = [
        run_stage_trace(
            sample=sample,
            config=experiment,
            llm=runtime.llm,
            client=qdrant_store.client,
            collection_name=qdrant_store.collection_name,
            embeddings=runtime.dense_embeddings,
            sparse_embeddings=runtime.sparse_embeddings,
            mongo_repository=storage_backend.mongo_repository,
            reranker=runtime.reranker,
        )
        for sample in samples
    ]
    metrics = {
        "retrieval": compute_ragas_retrieval_metrics(traces, llm=metrics_llm),
        "generation": compute_ragas_generation_metrics(
            traces,
            llm=metrics_llm,
            embeddings=metrics_embeddings,
        ),
    }
    artifact_paths = write_experiment_artifacts(
        output_dir=output_dir,
        experiment_name=experiment.experiment_name,
        traces=traces,
        metrics=metrics,
    )
    return {
        "experiment": experiment,
        "traces": traces,
        "metrics": metrics,
        "artifact_paths": artifact_paths,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.list_experiments:
            for experiment_name in DEFAULT_EXPERIMENT_NAMES:
                print(experiment_name)
            return 0
        if not args.dataset:
            parser.error("the following arguments are required: --dataset")

        runtime = build_eval_runtime()
        parent_chunk_records = load_parent_chunk_records(runtime.storage_backend.mongo_repository)
        samples = build_eval_samples(
            runtime=runtime,
            parent_chunk_records=parent_chunk_records,
            test_size=args.test_size,
        )
        write_eval_dataset(Path(args.dataset), samples)

        output_dir = Path(args.output_dir)
        for experiment in resolve_experiments(args.experiments):
            run_experiment(
                runtime=runtime,
                samples=samples,
                experiment=experiment,
                output_dir=output_dir,
            )
        return 0
    except SystemExit as exc:
        return int(exc.code)
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
