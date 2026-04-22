from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.data.schemas import IndexedItem
from src.models.base import EmbeddingModel
from src.retrieval.schemas import RetrievedItem, RetrievalResult
from src.retrieval.vector_store import VectorStore
from src.utils.io import read_jsonl


class RetrieverService:
    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        metadata_jsonl_path: str,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        rows = read_jsonl(path=Path(metadata_jsonl_path))
        self._items: dict[str, IndexedItem] = {str(r["item_id"]): IndexedItem(**r) for r in rows}

    @staticmethod
    def _passes_filters(item: IndexedItem, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        if "video_id" in filters and item.video_id != filters["video_id"]:
            return False
        if "modality" in filters and item.modality != filters["modality"]:
            return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        query_embedding = self.embedder.embed_texts([query])
        # Oversample before metadata filtering.
        raw_hits = self.vector_store.search(query_embedding, top_k=max(top_k * 5, top_k), filters=None)
        items: list[RetrievedItem] = []
        for hit in raw_hits:
            indexed = self._items.get(hit.item_id)
            if not indexed:
                continue
            if not self._passes_filters(indexed, filters):
                continue
            items.append(
                RetrievedItem(
                    item_id=indexed.item_id,
                    video_id=indexed.video_id,
                    modality=indexed.modality,
                    score=hit.score,
                    timestamp_start=indexed.timestamp_start,
                    timestamp_end=indexed.timestamp_end,
                    text=indexed.text,
                    frame_path=indexed.frame_path,
                    source_id=indexed.source_id,
                )
            )
            if len(items) >= top_k:
                break
        return RetrievalResult(query=query, items=items)

    def retrieve_debug_dict(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Helper for quick CLI/debug inspection.
        """
        result = self.retrieve(query=query, top_k=top_k, filters=filters)
        return {"query": result.query, "items": [asdict(x) for x in result.items]}

    # TODO: add agentic helpers:
    # - lookup_nearby_chunks(item_id, seconds_before, seconds_after)
    # - lookup_time_window(video_id, start_sec, end_sec)
