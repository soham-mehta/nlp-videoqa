from __future__ import annotations

from pathlib import Path

import modal

from src.config.settings import AppConfig, EmbeddingConfig, PathsConfig
from src.indexing.build import build_index_for_video_ids

APP_NAME = "nlp-videoqa-index-builder"
INDEX_VOLUME_NAME = "nlp-videoqa-index"
ASSETS_VOLUME_NAME = "nlp-videoqa-assets"
HF_SECRET_NAME = "huggingface-secret"
ASSETS_MOUNT_PATH = "/vol/assets"
INDEX_MOUNT_PATH = "/vol/index"
HF_CACHE_PATH = "/root/.cache/huggingface"
GPU = "L4"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "faiss-cpu>=1.8.0",
        "numpy>=2.0.0",
        "Pillow>=10.4.0",
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "huggingface_hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME)
index_volume = modal.Volume.from_name(INDEX_VOLUME_NAME, create_if_missing=True)
assets_volume = modal.Volume.from_name(ASSETS_VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4.0,
    memory=32768,
    timeout=4 * 60 * 60,
    scaledown_window=900,
    secrets=[hf_secret],
    volumes={
        ASSETS_MOUNT_PATH: assets_volume,
        INDEX_MOUNT_PATH: index_volume,
        HF_CACHE_PATH: hf_cache,
    },
)
def build_remote_index(
    *,
    video_ids: list[str],
    index_subdir: str = "indexes/default",
    embedding_model_name: str = "google/siglip2-base-patch16-224",
    batch_size: int = 16,
) -> dict[str, object]:
    repo_root = Path(ASSETS_MOUNT_PATH)
    index_dir = Path(INDEX_MOUNT_PATH) / index_subdir
    index_dir.mkdir(parents=True, exist_ok=True)

    paths = PathsConfig(
        repo_root=repo_root,
        transcripts_dir=repo_root / "data" / "transcripts",
        frames_dir=repo_root / "data" / "frames",
        frames_metadata_dir=repo_root / "data" / "metadata" / "frames",
        index_dir=index_dir,
        metadata_jsonl_path=index_dir / "items.jsonl",
        faiss_index_path=index_dir / "vectors.faiss",
        faiss_ids_path=index_dir / "vector_ids.json",
    )
    cfg = AppConfig(
        paths=paths,
        indexing=AppConfig.default(repo_root=repo_root).indexing,
        retrieval=AppConfig.default(repo_root=repo_root).retrieval,
        generation=AppConfig.default(repo_root=repo_root).generation,
        embedding=EmbeddingConfig(
            model_name=embedding_model_name,
            device="cuda",
            batch_size=batch_size,
            normalize=True,
        ),
    )
    build_index_for_video_ids(cfg, sorted(set(video_ids)))
    index_volume.commit()
    return {
        "status": "ok",
        "video_ids": sorted(set(video_ids)),
        "index_subdir": index_subdir,
        "metadata_jsonl_path": str(paths.metadata_jsonl_path),
        "faiss_index_path": str(paths.faiss_index_path),
        "faiss_ids_path": str(paths.faiss_ids_path),
    }
