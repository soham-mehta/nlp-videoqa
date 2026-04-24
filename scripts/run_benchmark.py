from __future__ import annotations

# ruff: noqa: E402
# Repo root must be on sys.path before `src` imports (script may be run without PYTHONPATH=.).
import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config.settings import AppConfig, EmbeddingConfig, GenerationConfig
from src.eval.benchmark_runner import append_benchmark_progress, run_benchmark
from src.eval.reporting import save_benchmark_run
from src.utils.io import write_jsonl
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
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=Path("data/eval/predictions_v1.jsonl"),
        help="Strict prediction_v1 JSONL for grading and cross-system comparison.",
    )
    parser.add_argument(
        "--detail-jsonl",
        type=Path,
        default=None,
        help="Optional verbose per-question JSONL (metrics + debug). Omit to skip.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (e.g. for CI logs).",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("data/eval/benchmark_progress.txt"),
        help="Append-only log of question START/DONE (use: tail -f this file).",
    )
    parser.add_argument(
        "--no-progress-file",
        action="store_true",
        help="Do not write --progress-file.",
    )
    args = parser.parse_args()

    progress_path: str | None = None
    if not args.no_progress_file and args.progress_file is not None:
        pf = args.progress_file.resolve()
        pf.parent.mkdir(parents=True, exist_ok=True)
        progress_path = str(pf)
        pf.write_text(
            f"# benchmark run started {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        append_benchmark_progress(
            progress_path,
            "[BOOT] loading SigLIP embedder, FAISS index, Qwen VL (this can take several minutes)…",
        )

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
    if progress_path:
        append_benchmark_progress(progress_path, "[BOOT] models ready; starting benchmark questions")

    try:
        result = run_benchmark(
            rag_service=rag,
            benchmark_path=str(args.benchmark_path),
            top_k_total=args.top_k_total,
            max_text_items=args.max_text_items,
            max_frame_items=args.max_frame_items,
            system_name=args.system_name,
            show_progress=not args.no_progress,
            progress_file=progress_path,
        )
    except BaseException as exc:
        if progress_path:
            append_benchmark_progress(
                progress_path,
                f"[RUN_ABORTED] {type(exc).__name__}: {exc}",
            )
        raise
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
    if progress_path:
        append_benchmark_progress(
            progress_path,
            "[RUN_ALL_DONE] wrote predictions and benchmark_run json",
        )


if __name__ == "__main__":
    main()
