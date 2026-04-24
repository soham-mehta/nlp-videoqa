from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config.settings import AppConfig, EmbeddingConfig
from src.models.siglip_embedder import SigLIP2EmbeddingModel
from src.retrieval.faiss_store import FaissVectorStore
from src.retrieval.service import RetrieverService


def main() -> None:
    parser = argparse.ArgumentParser(description="Query multimodal FAISS index.")
    parser.add_argument("query", type=str)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-name", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--modality", type=str, choices=["text", "frame"], default=None)
    args = parser.parse_args()

    cfg = AppConfig.default(repo_root=args.repo_root)
    embedder = SigLIP2EmbeddingModel(
        EmbeddingConfig(
            model_name=args.model_name,
            device=args.device,
            batch_size=16,
            normalize=True,
        )
    )
    store = FaissVectorStore.load(cfg.paths.faiss_index_path, cfg.paths.faiss_ids_path)
    retriever = RetrieverService(
        embedder=embedder,
        vector_store=store,
        metadata_jsonl_path=str(cfg.paths.metadata_jsonl_path),
    )
    filters = {}
    if args.video_id:
        filters["video_id"] = args.video_id
    if args.modality:
        filters["modality"] = args.modality
    output = retriever.retrieve_debug_dict(args.query, top_k=args.top_k, filters=filters or None)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
