from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.agentic.answering import AgenticRAGAnsweringService
from src.agentic.modal_backend import ModalAgenticBackend
from src.config.settings import AppConfig, EmbeddingConfig, GenerationConfig
from src.eval.benchmark_runner import run_benchmark
from src.eval.reporting import save_benchmark_run
from src.models.siglip_embedder import SigLIP2EmbeddingModel
from src.retrieval.modal_client import ModalRetrievalService
from src.retrieval.faiss_store import FaissVectorStore
from src.retrieval.service import RetrieverService
from src.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentic multimodal RAG benchmark.")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/benchmark/example_benchmark_v1.jsonl"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--top-k-total", type=int, default=8)
    parser.add_argument("--max-text-items", type=int, default=4)
    parser.add_argument("--max-frame-items", type=int, default=4)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--embedding-model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--retrieval-backend", type=str, choices=["local", "modal"], default="local")
    parser.add_argument("--modal-retrieval-app-name", type=str, default="nlp-videoqa-retrieval")
    parser.add_argument("--modal-retrieval-class-name", type=str, default="RetrievalIndex")
    parser.add_argument("--modal-index-subdir", type=str, default="indexes/default")
    parser.add_argument("--generation-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation-base-url", type=str, default=None)
    parser.add_argument("--generation-api-key", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--system-name", type=str, default="agentic_rag")
    parser.add_argument("--output-json", type=Path, default=Path("data/eval/agentic_benchmark_run.json"))
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=Path("data/eval/agentic_predictions_v1.jsonl"),
        help="Strict prediction_v1 JSONL for grading and cross-system comparison.",
    )
    parser.add_argument(
        "--detail-jsonl",
        type=Path,
        default=None,
        help="Optional verbose per-question JSONL (metrics + debug). Omit to skip.",
    )
    args = parser.parse_args()

    cfg = AppConfig.default(repo_root=args.repo_root)
    if args.retrieval_backend == "local":
        embedder = SigLIP2EmbeddingModel(
            EmbeddingConfig(model_name=args.embedding_model, device=args.device, batch_size=16, normalize=True)
        )
        store = FaissVectorStore.load(cfg.paths.faiss_index_path, cfg.paths.faiss_ids_path)
        retriever = RetrieverService(
            embedder=embedder,
            vector_store=store,
            metadata_jsonl_path=str(cfg.paths.metadata_jsonl_path),
        )
        backend = None
        metadata_jsonl_path = str(cfg.paths.metadata_jsonl_path)
    else:
        retriever = ModalRetrievalService(
            app_name=args.modal_retrieval_app_name,
            class_name=args.modal_retrieval_class_name,
            embedding_model_name=args.embedding_model,
            index_subdir=args.modal_index_subdir,
        )
        backend = ModalAgenticBackend(retrieval_service=retriever)
        metadata_jsonl_path = None
    answerer = AgenticRAGAnsweringService(
        retriever=retriever,
        generation_config=GenerationConfig(
            model_name=args.generation_model,
            device=args.device,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            do_sample=cfg.generation.do_sample,
            base_url=args.generation_base_url or cfg.generation.base_url,
            api_key=args.generation_api_key or cfg.generation.api_key,
            timeout_sec=cfg.generation.timeout_sec,
        ),
        metadata_jsonl_path=metadata_jsonl_path,
        backend=backend,
        max_tool_calls=args.max_tool_calls,
    )

    result = run_benchmark(
        rag_service=answerer,
        benchmark_path=str(args.benchmark_path),
        top_k_total=args.top_k_total,
        max_text_items=args.max_text_items,
        max_frame_items=args.max_frame_items,
        system_name=args.system_name,
    )
    run_dict = asdict(result)
    prediction_rows = run_dict.get("prediction_rows", [])
    if args.predictions_jsonl is not None:
        write_jsonl(args.predictions_jsonl, prediction_rows)
    save_benchmark_run(args.output_json, args.detail_jsonl, run_dict, run_dict.get("per_question", []))
    print(json.dumps(run_dict, indent=2, ensure_ascii=False))
    if args.predictions_jsonl is not None:
        print(f"\nSaved strict predictions (prediction_v1) to: {args.predictions_jsonl}")
    if args.detail_jsonl is not None:
        print(f"Saved detail per-question rows to: {args.detail_jsonl}")


if __name__ == "__main__":
    main()
