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
        self._item_count = len(self._items)

    @staticmethod
    def _passes_filters(item: IndexedItem, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        if "video_id" in filters and item.video_id != filters["video_id"]:
            return False
        if "modality" in filters and item.modality != filters["modality"]:
            return False
        return True

    @staticmethod
    def _modality_targets(top_k: int) -> dict[str, int]:
        """
        For mixed retrieval, reserve slots for both modalities.
        """
        if top_k < 2:
            return {"text": top_k, "frame": 0}
        remaining = top_k - 2
        return {
            "text": 1 + (remaining // 2) + (remaining % 2),
            "frame": 1 + (remaining // 2),
        }

    def _as_retrieved_item(self, item: IndexedItem, score: float) -> RetrievedItem:
        return RetrievedItem(
            item_id=item.item_id,
            video_id=item.video_id,
            modality=item.modality,
            score=score,
            timestamp_start=item.timestamp_start,
            timestamp_end=item.timestamp_end,
            text=item.text,
            frame_path=item.frame_path,
            source_id=item.source_id,
        )

    def _search_candidates(self, query_embedding: Any, top_k: int, exhaustive: bool = False) -> list[RetrievedItem]:
        candidate_k = self._item_count if exhaustive else min(self._item_count, max(top_k * 20, 200))
        raw_hits = self.vector_store.search(query_embedding, top_k=candidate_k, filters=None)
        out: list[RetrievedItem] = []
        for hit in raw_hits:
            indexed = self._items.get(hit.item_id)
            if not indexed:
                continue
            out.append(self._as_retrieved_item(indexed, hit.score))
        return out

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        query_embedding = self.embedder.embed_texts([query])
        exhaustive = not (filters and "modality" in filters)
        candidates = self._search_candidates(query_embedding=query_embedding, top_k=top_k, exhaustive=exhaustive)

        filtered = [item for item in candidates if self._passes_filters(self._items[item.item_id], filters)]

        # If user asks for a single modality, respect it directly.
        if filters and "modality" in filters:
            return RetrievalResult(query=query, items=filtered[:top_k])

        # Default behavior for multimodal RAG: try to include both text and frame evidence.
        targets = self._modality_targets(top_k=top_k)
        selected: list[RetrievedItem] = []
        used_ids: set[str] = set()
        counts = {"text": 0, "frame": 0}

        for item in filtered:
            if item.item_id in used_ids:
                continue
            if counts.get(item.modality, 0) >= targets.get(item.modality, 0):
                continue
            selected.append(item)
            used_ids.add(item.item_id)
            counts[item.modality] = counts.get(item.modality, 0) + 1
            if len(selected) >= top_k:
                break

        # Fill any remaining slots with best-scoring leftovers regardless of modality.
        if len(selected) < top_k:
            for item in filtered:
                if item.item_id in used_ids:
                    continue
                selected.append(item)
                used_ids.add(item.item_id)
                if len(selected) >= top_k:
                    break
        return RetrievalResult(query=query, items=selected)

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
