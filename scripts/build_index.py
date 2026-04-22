from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import AppConfig, EmbeddingConfig
from src.indexing.build import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal FAISS index.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-name", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cfg = AppConfig.default(repo_root=args.repo_root)
    cfg = AppConfig(
        paths=cfg.paths,
        indexing=cfg.indexing,
        retrieval=cfg.retrieval,
        generation=cfg.generation,
        embedding=EmbeddingConfig(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            normalize=True,
        ),
    )
    build_index(cfg)


if __name__ == "__main__":
    main()
