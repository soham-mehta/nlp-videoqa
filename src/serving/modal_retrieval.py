from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from src.config.settings import EmbeddingConfig
from src.models.siglip_embedder import SigLIP2EmbeddingModel
from src.retrieval.faiss_store import FaissVectorStore
from src.retrieval.service import RetrieverService
from src.utils.io import read_jsonl

APP_NAME = "nlp-videoqa-retrieval"
INDEX_VOLUME_NAME = "nlp-videoqa-index"
HF_SECRET_NAME = "huggingface-secret"
INDEX_MOUNT_PATH = "/vol/index"
HF_CACHE_PATH = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.11")
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
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


@app.cls(
    image=image,
    cpu=4.0,
    memory=16384,
    scaledown_window=900,
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes={
        INDEX_MOUNT_PATH: index_volume,
        HF_CACHE_PATH: hf_cache,
    },
)
class RetrievalIndex:
    embedding_model_name: str = modal.parameter(default="google/siglip2-base-patch16-224")
    index_subdir: str = modal.parameter(default="indexes/default")

    @modal.enter()
    def load(self) -> None:
        index_dir = Path(INDEX_MOUNT_PATH) / self.index_subdir
        faiss_index_path = index_dir / "vectors.faiss"
        faiss_ids_path = index_dir / "vector_ids.json"
        metadata_jsonl_path = index_dir / "items.jsonl"

        if not faiss_index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {faiss_index_path}")
        if not faiss_ids_path.exists():
            raise FileNotFoundError(f"Missing vector id file: {faiss_ids_path}")
        if not metadata_jsonl_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_jsonl_path}")

        self.embedder = SigLIP2EmbeddingModel(
            EmbeddingConfig(
                model_name=self.embedding_model_name,
                device="cpu",
                batch_size=16,
                normalize=True,
            )
        )
        self.vector_store = FaissVectorStore.load(faiss_index_path, faiss_ids_path)
        self.retriever = RetrieverService(
            embedder=self.embedder,
            vector_store=self.vector_store,
            metadata_jsonl_path=str(metadata_jsonl_path),
        )
        rows = read_jsonl(metadata_jsonl_path)
        self.items_by_id: dict[str, dict[str, Any]] = {str(row["item_id"]): row for row in rows}
        self.index_info = {
            "embedding_model_name": self.embedding_model_name,
            "index_subdir": self.index_subdir,
            "num_items": len(rows),
            "vector_dim": self.vector_store.dim,
        }

    @modal.method()
    def healthcheck(self) -> dict[str, Any]:
        return {"status": "ok", **self.index_info}

    @modal.method()
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.retriever.retrieve(query=query, top_k=top_k, filters=filters)
        return {"query": result.query, "items": [asdict(item) for item in result.items]}

    @modal.method()
    def semantic_search(
        self,
        query: str,
        k: int = 8,
        modality: str | None = None,
        video_id: str | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {}
        if video_id is not None:
            filters["video_id"] = video_id
        if modality is not None:
            filters["modality"] = modality
        result = self.retriever.retrieve(
            query=query,
            top_k=k,
            filters=filters or None,
        )
        items = result.items
        if t_start is not None or t_end is not None:
            items = [
                item
                for item in items
                if (t_start is None or item.timestamp_end >= t_start)
                and (t_end is None or item.timestamp_start <= t_end)
            ]
        return [asdict(item) for item in items[:k]]

    @modal.method()
    def get_chunks_by_timestamp(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        modality: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.items_by_id.values()
            if row["video_id"] == video_id
            and float(row["timestamp_end"]) >= t_start
            and float(row["timestamp_start"]) <= t_end
            and (modality is None or row["modality"] == modality)
        ]
        rows.sort(key=lambda row: (float(row["timestamp_start"]), str(row["item_id"])))
        return rows

    @modal.method()
    def get_nearby_chunks(self, chunk_id: str, radius_seconds: float = 10.0) -> list[dict[str, Any]]:
        anchor = self.items_by_id.get(chunk_id)
        if anchor is None:
            return []
        lo = float(anchor["timestamp_start"]) - radius_seconds
        hi = float(anchor["timestamp_end"]) + radius_seconds
        rows = [
            row
            for row in self.items_by_id.values()
            if row["video_id"] == anchor["video_id"]
            and float(row["timestamp_end"]) >= lo
            and float(row["timestamp_start"]) <= hi
        ]
        rows.sort(key=lambda row: (float(row["timestamp_start"]), str(row["item_id"])))
        return rows

    @modal.method()
    def get_video_metadata(self, video_id: str) -> dict[str, Any]:
        rows = [row for row in self.items_by_id.values() if row["video_id"] == video_id]
        if not rows:
            return {}
        modality_counts: dict[str, int] = {}
        for row in rows:
            modality = str(row["modality"])
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        return {
            "video_id": video_id,
            "duration_sec": max(float(row["timestamp_end"]) for row in rows),
            "modalities": sorted(modality_counts),
            "counts_by_modality": modality_counts,
            "num_indexed_items": len(rows),
        }

    @modal.method()
    def resolve_chunks(self, item_ids: list[str]) -> list[dict[str, Any]]:
        return [self.items_by_id[item_id] for item_id in item_ids if item_id in self.items_by_id]
