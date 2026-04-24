from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config.settings import AppConfig, EmbeddingConfig, GenerationConfig
from src.eval.reporting import save_answer_run
from src.models.siglip_embedder import SigLIP2EmbeddingModel
from src.rag.answering import BaselineRAGAnsweringService
from src.rag.openai_generator import OpenAIChatGenerator
from src.rag.schemas import AnswerRequest, RetrievalPolicy
from src.retrieval.modal_client import ModalRetrievalService
from src.retrieval.faiss_store import FaissVectorStore
from src.retrieval.service import RetrieverService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline multimodal RAG for one question.")
    parser.add_argument("question", type=str)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--top-k-total", type=int, default=8)
    parser.add_argument("--max-text-items", type=int, default=4)
    parser.add_argument("--max-frame-items", type=int, default=4)
    parser.add_argument("--embedding-model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--retrieval-backend", type=str, choices=["local", "modal"], default="local")
    parser.add_argument("--modal-retrieval-app-name", type=str, default="nlp-videoqa-retrieval")
    parser.add_argument("--modal-retrieval-class-name", type=str, default="RetrievalIndex")
    parser.add_argument("--modal-index-subdir", type=str, default="indexes/default")
    parser.add_argument("--generation-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generation-base-url", type=str, default=None)
    parser.add_argument("--generation-api-key", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
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
    else:
        retriever = ModalRetrievalService(
            app_name=args.modal_retrieval_app_name,
            class_name=args.modal_retrieval_class_name,
            embedding_model_name=args.embedding_model,
            index_subdir=args.modal_index_subdir,
        )
    generator = OpenAIChatGenerator(
        GenerationConfig(
            model_name=args.generation_model,
            device=args.device,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            do_sample=cfg.generation.do_sample,
            base_url=args.generation_base_url or cfg.generation.base_url,
            api_key=args.generation_api_key or cfg.generation.api_key,
            timeout_sec=cfg.generation.timeout_sec,
        )
    )
    rag = BaselineRAGAnsweringService(retriever=retriever, generator=generator)
    request = AnswerRequest(
        question=args.question,
        video_id=args.video_id,
        retrieval_policy=RetrievalPolicy(
            top_k_total=args.top_k_total,
            max_text_items=args.max_text_items,
            max_frame_items=args.max_frame_items,
        ),
    )
    result = rag.run(request)
    result_dict = asdict(result)
    output_path = args.output_json
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data/runs/answers") / f"answer_run_{stamp}.json"
    save_answer_run(output_path, result_dict)
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    print(f"\nSaved run output to: {output_path}")


if __name__ == "__main__":
    main()
