from __future__ import annotations

from typing import Any

from src.agentic.backend import Chunk
from src.retrieval.modal_client import ModalRetrievalService


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        item_id=str(row["item_id"]),
        video_id=str(row["video_id"]),
        modality=str(row["modality"]),  # type: ignore[arg-type]
        timestamp_start=float(row["timestamp_start"]),
        timestamp_end=float(row["timestamp_end"]),
        text=row.get("text"),
        frame_path=row.get("frame_path"),
        source_id=row.get("source_id"),
        score=float(row["score"]) if row.get("score") is not None else None,
    )


class ModalAgenticBackend:
    def __init__(self, retrieval_service: ModalRetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def semantic_search(
        self,
        query: str,
        k: int = 8,
        modality: str | None = None,
        video_id: str | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> list[Chunk]:
        rows = self.retrieval_service.semantic_search(
            query=query,
            k=k,
            modality=modality,
            video_id=video_id,
            t_start=t_start,
            t_end=t_end,
        )
        return [_row_to_chunk(row) for row in rows]

    def get_chunks_by_timestamp(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        modality: str | None = None,
    ) -> list[Chunk]:
        rows = self.retrieval_service.get_chunks_by_timestamp(
            video_id=video_id,
            t_start=t_start,
            t_end=t_end,
            modality=modality,
        )
        return [_row_to_chunk(row) for row in rows]

    def get_nearby_chunks(self, chunk_id: str, radius_seconds: float = 10.0) -> list[Chunk]:
        rows = self.retrieval_service.get_nearby_chunks(
            chunk_id=chunk_id,
            radius_seconds=radius_seconds,
        )
        return [_row_to_chunk(row) for row in rows]

    def get_video_metadata(self, video_id: str) -> dict[str, Any]:
        return self.retrieval_service.get_video_metadata(video_id=video_id)

    def resolve_chunks(self, item_ids: list[str]) -> list[Chunk]:
        rows = self.retrieval_service.resolve_chunks(item_ids=item_ids)
        return [_row_to_chunk(row) for row in rows]
