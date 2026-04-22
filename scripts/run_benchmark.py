from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.config.settings import AppConfig, EmbeddingConfig, GenerationConfig
from src.eval.benchmark_runner import run_benchmark
from src.eval.reporting import save_benchmark_run
from src.models.siglip_embedder import SigLIP2EmbeddingModel
from src.rag.answering import BaselineRAGAnsweringService
from src.rag.qwen_vl_generator import QwenVLGenerator
from src.retrieval.faiss_store import FaissVectorStore
from src.retrieval.service import RetrieverService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline multimodal RAG benchmark.")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/benchmark/example_benchmark_v1.jsonl"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--top-k-total", type=int, default=8)
    parser.add_argument("--max-text-items", type=int, default=4)
    parser.add_argument("--max-frame-items", type=int, default=4)
    parser.add_argument("--embedding-model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--generation-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--system-name", type=str, default="baseline_rag")
    parser.add_argument("--output-json", type=Path, default=Path("data/eval/benchmark_run.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/eval/benchmark_questions.jsonl"))
    args = parser.parse_args()

    cfg = AppConfig.default(repo_root=args.repo_root)
    embedder = SigLIP2EmbeddingModel(
        EmbeddingConfig(model_name=args.embedding_model, device=args.device, batch_size=16, normalize=True)
    )
    store = FaissVectorStore.load(cfg.paths.faiss_index_path, cfg.paths.faiss_ids_path)
    retriever = RetrieverService(
        embedder=embedder,
        vector_store=store,
        metadata_jsonl_path=str(cfg.paths.metadata_jsonl_path),
    )
    generator = QwenVLGenerator(
        GenerationConfig(
            model_name=args.generation_model,
            device=args.device,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            do_sample=cfg.generation.do_sample,
        )
    )
    rag = BaselineRAGAnsweringService(retriever=retriever, generator=generator)

    result = run_benchmark(
        rag_service=rag,
        benchmark_path=str(args.benchmark_path),
        top_k_total=args.top_k_total,
        max_text_items=args.max_text_items,
        max_frame_items=args.max_frame_items,
        system_name=args.system_name,
    )
    run_dict = asdict(result)
    per_question_rows = run_dict.get("per_question", [])
    save_benchmark_run(args.output_json, args.output_jsonl, run_dict, per_question_rows)
    print(json.dumps(run_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
