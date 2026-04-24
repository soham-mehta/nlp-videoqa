from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.config.settings import AppConfig
from src.data.loaders import build_indexed_items_from_video, discover_video_ids
from src.data.schemas import IndexedItem
from src.models.base import EmbeddingModel
from src.models.siglip_embedder import SigLIP2EmbeddingModel
from src.retrieval.faiss_store import FaissVectorStore
from src.utils.io import write_jsonl
from src.utils.logging import get_logger

logger = get_logger("indexing.build")


def _embed_items(items: list[IndexedItem], embedder: EmbeddingModel) -> tuple[list[str], np.ndarray]:
    text_items = [x for x in items if x.modality == "text" and x.text]
    frame_items = [x for x in items if x.modality == "frame" and x.frame_path]

    text_vectors = embedder.embed_texts([x.text or "" for x in text_items]) if text_items else None
    images: list[Image.Image] = []
    image_item_ids: list[str] = []
    for x in frame_items:
        try:
            images.append(Image.open(Path(x.frame_path)).convert("RGB"))
            image_item_ids.append(x.item_id)
        except FileNotFoundError:
            logger.warning("Missing frame path, skipping: %s", x.frame_path)
    frame_vectors = embedder.embed_images(images) if images else None

    vectors_by_id: dict[str, np.ndarray] = {}
    if text_vectors is not None:
        for item, vec in zip(text_items, text_vectors):
            vectors_by_id[item.item_id] = vec
    if frame_vectors is not None:
        for item_id, vec in zip(image_item_ids, frame_vectors):
            vectors_by_id[item_id] = vec

    ordered_ids: list[str] = []
    ordered_vectors: list[np.ndarray] = []
    for item in items:
        vec = vectors_by_id.get(item.item_id)
        if vec is None:
            continue
        ordered_ids.append(item.item_id)
        ordered_vectors.append(vec)
    if not ordered_vectors:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ordered_ids, np.vstack(ordered_vectors).astype(np.float32)


def build_index_for_video_ids(config: AppConfig, video_ids: list[str]) -> None:
    all_items: list[IndexedItem] = []
    for video_id in video_ids:
        all_items.extend(
            build_indexed_items_from_video(
                paths=config.paths,
                video_id=video_id,
                frame_fps=config.indexing.frame_sample_fps,
            )
        )

    if not all_items:
        raise RuntimeError("No indexed items built from transcripts/frames")
    logger.info("Built %d indexed items across %d videos", len(all_items), len(video_ids))

    embedder = SigLIP2EmbeddingModel(config.embedding)
    item_ids, embeddings = _embed_items(all_items, embedder=embedder)
    if embeddings.shape[0] == 0:
        raise RuntimeError("No embeddings produced")

    item_by_id = {x.item_id: x for x in all_items}
    kept_items = [item_by_id[item_id] for item_id in item_ids]
    store = FaissVectorStore(dim=embeddings.shape[1])
    store.add(item_ids, embeddings)

    write_jsonl(config.paths.metadata_jsonl_path, [asdict(x) for x in kept_items])
    store.save(config.paths.faiss_index_path, config.paths.faiss_ids_path)
    logger.info("Wrote metadata: %s", config.paths.metadata_jsonl_path)
    logger.info("Wrote Faiss index: %s", config.paths.faiss_index_path)


def build_index(config: AppConfig) -> None:
    video_ids = discover_video_ids(config.paths)
    if not video_ids:
        raise RuntimeError(f"No transcript files found in {config.paths.transcripts_dir}")
    build_index_for_video_ids(config, video_ids)
