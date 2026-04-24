from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    repo_root: Path
    transcripts_dir: Path
    frames_dir: Path
    frames_metadata_dir: Path
    index_dir: Path
    metadata_jsonl_path: Path
    faiss_index_path: Path
    faiss_ids_path: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "PathsConfig":
        data_dir = repo_root / "data"
        index_dir = data_dir / "indexes" / "default"
        return PathsConfig(
            repo_root=repo_root,
            transcripts_dir=data_dir / "transcripts",
            frames_dir=data_dir / "frames",
            frames_metadata_dir=data_dir / "metadata" / "frames",
            index_dir=index_dir,
            metadata_jsonl_path=index_dir / "items.jsonl",
            faiss_index_path=index_dir / "vectors.faiss",
            faiss_ids_path=index_dir / "vector_ids.json",
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "google/siglip2-base-patch16-224"
    device: str = "cpu"
    batch_size: int = 16
    normalize: bool = True


@dataclass(frozen=True)
class IndexConfig:
    frame_sample_fps: float = 1.0
    text_modality_name: str = "text"
    frame_modality_name: str = "frame"


@dataclass(frozen=True)
class RetrievalPolicyConfig:
    top_k_total: int = 8
    max_text_items: int = 4
    max_frame_items: int = 4
    dedupe_seconds: float = 1.0


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.2
    do_sample: bool = False
    base_url: str = field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("VLLM_API_KEY", "EMPTY"))
    timeout_sec: float = 120.0


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    embedding: EmbeddingConfig
    indexing: IndexConfig
    retrieval: RetrievalPolicyConfig
    generation: GenerationConfig

    @staticmethod
    def default(repo_root: Path | None = None) -> "AppConfig":
        root = repo_root or Path(__file__).resolve().parents[2]
        return AppConfig(
            paths=PathsConfig.from_repo_root(root),
            embedding=EmbeddingConfig(),
            indexing=IndexConfig(),
            retrieval=RetrievalPolicyConfig(),
            generation=GenerationConfig(),
        )
