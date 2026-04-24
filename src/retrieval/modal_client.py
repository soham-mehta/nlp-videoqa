from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import modal

from src.retrieval.schemas import RetrievedItem, RetrievalResult

DEFAULT_MODAL_RETRIEVAL_APP_NAME = "nlp-videoqa-retrieval"
DEFAULT_MODAL_RETRIEVAL_CLASS_NAME = "RetrievalIndex"


class ChunkResolver(Protocol):
    def resolve_chunks(self, item_ids: list[str]) -> list[dict[str, Any]]: ...


def _row_to_retrieved_item(row: dict[str, Any]) -> RetrievedItem:
    return RetrievedItem(
        item_id=str(row["item_id"]),
        video_id=str(row["video_id"]),
        modality=str(row["modality"]),  # type: ignore[arg-type]
        score=float(row.get("score", 0.0)),
        timestamp_start=float(row["timestamp_start"]),
        timestamp_end=float(row["timestamp_end"]),
        text=row.get("text"),
        frame_path=row.get("frame_path"),
        source_id=row.get("source_id"),
    )


class ModalRetrievalService:
    def __init__(
        self,
        *,
        app_name: str = DEFAULT_MODAL_RETRIEVAL_APP_NAME,
        class_name: str = DEFAULT_MODAL_RETRIEVAL_CLASS_NAME,
        embedding_model_name: str = "google/siglip2-base-patch16-224",
        index_subdir: str = "indexes/default",
    ) -> None:
        cls = modal.Cls.from_name(app_name, class_name)
        self._instance = cls(
            embedding_model_name=embedding_model_name,
            index_subdir=index_subdir,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        row = self._instance.retrieve.remote(query=query, top_k=top_k, filters=filters)
        return RetrievalResult(
            query=str(row["query"]),
            items=[_row_to_retrieved_item(item) for item in row.get("items", [])],
        )

    def retrieve_debug_dict(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.retrieve(query=query, top_k=top_k, filters=filters)
        return {"query": result.query, "items": [asdict(item) for item in result.items]}

    def semantic_search(
        self,
        query: str,
        k: int = 8,
        modality: str | None = None,
        video_id: str | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._instance.semantic_search.remote(
            query=query,
            k=k,
            modality=modality,
            video_id=video_id,
            t_start=t_start,
            t_end=t_end,
        )

    def get_chunks_by_timestamp(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        modality: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._instance.get_chunks_by_timestamp.remote(
            video_id=video_id,
            t_start=t_start,
            t_end=t_end,
            modality=modality,
        )

    def get_nearby_chunks(self, chunk_id: str, radius_seconds: float = 10.0) -> list[dict[str, Any]]:
        return self._instance.get_nearby_chunks.remote(
            chunk_id=chunk_id,
            radius_seconds=radius_seconds,
        )

    def get_video_metadata(self, video_id: str) -> dict[str, Any]:
        return self._instance.get_video_metadata.remote(video_id=video_id)

    def resolve_chunks(self, item_ids: list[str]) -> list[dict[str, Any]]:
        return self._instance.resolve_chunks.remote(item_ids=item_ids)


def upload_index_to_modal_volume(
    *,
    volume_name: str,
    local_index_dir: Path,
    remote_index_subdir: str = "indexes/default",
) -> None:
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(local_index_dir / "vectors.faiss", f"/{remote_index_subdir}/vectors.faiss")
        batch.put_file(local_index_dir / "vector_ids.json", f"/{remote_index_subdir}/vector_ids.json")
        batch.put_file(local_index_dir / "items.jsonl", f"/{remote_index_subdir}/items.jsonl")
