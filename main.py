from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

from rag_demo.chunking import ChunkingConfig
from rag_demo.embeddings import CachedEmbeddings
from rag_demo.pipeline import PipelineConfig, run_batch_pipeline, run_document_pipeline
from rag_demo.storage import build_storage_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the document preprocessing demo.")
    parser.add_argument("--file", help="Path to the file to process.")
    parser.add_argument("--data-dir", help="Directory containing supported files to process in batch.")
    return parser


def _print_summary(result) -> None:
    print(f"Status: {result.status}")
    if result.content_hash:
        print(f"Content hash: {result.content_hash}")
    if result.skip_reason:
        print(f"Skip reason: {result.skip_reason}")
    print(f"Raw pages: {result.raw_page_count}")
    print(f"Cleaned pages: {result.cleaned_page_count}")
    print(f"Parent chunks: {len(result.parent_chunks)}")
    print(f"Child chunks: {len(result.child_chunks)}")
    if result.parent_chunks:
        print(f"Sample parent metadata: {result.parent_chunks[0].metadata}")
    if result.child_chunks:
        print(f"Sample child metadata: {result.child_chunks[0].metadata}")


def _print_batch_summary(result) -> None:
    print(f"Discovered files: {result.total_files}")
    print(f"Successful: {result.successful_files}")
    print(f"Skipped: {result.skipped_files}")
    print(f"Failed: {result.failed_files}")
    for file_result in result.files:
        file_name = file_result.file_path.name
        if file_result.status == "ok":
            print(
                f"{file_name} raw={file_result.raw_page_count} cleaned={file_result.cleaned_page_count} "
                f"parent={file_result.parent_chunk_count} child={file_result.child_chunk_count}"
            )
        elif file_result.status == "skipped":
            print(f"{file_name} skipped reason={file_result.skip_reason}")
        else:
            print(f"{file_name} error={file_result.error}")


def _build_embeddings() -> CachedEmbeddings:
    return CachedEmbeddings(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )


def _build_sparse_embeddings():
    return SparseTextEmbedding(model_name="Qdrant/bm25")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.file) == bool(args.data_dir):
        parser.error("choose exactly one of --file or --data-dir")

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        embeddings = _build_embeddings()
        sparse_embeddings = _build_sparse_embeddings()
        storage_backend = build_storage_backend(sparse_embeddings=sparse_embeddings)
        result = run_document_pipeline(
            file_path,
            PipelineConfig(chunking=ChunkingConfig()),
            embeddings,
            storage_backend=storage_backend,
        )
        _print_summary(result)
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
        if not data_dir.is_dir():
            raise NotADirectoryError(data_dir)
        embeddings = _build_embeddings()
        sparse_embeddings = _build_sparse_embeddings()
        storage_backend = build_storage_backend(sparse_embeddings=sparse_embeddings)
        result = run_batch_pipeline(
            data_dir,
            embeddings,
            storage_backend=storage_backend,
        )
        _print_batch_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
