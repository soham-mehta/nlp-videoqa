from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.agentic.backend import Chunk, Modality
from src.data.schemas import IndexedItem
from src.models.base import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.utils.io import read_jsonl


def _interval_overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return (a_start < b_end) and (b_start < a_end)


class BenchmarkRetrievalBackend:
    """
    Tool backend over the same FAISS index + metadata used by the baseline benchmark.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        metadata_jsonl_path: str,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        rows = read_jsonl(Path(metadata_jsonl_path))
        self._items: dict[str, IndexedItem] = {str(row["item_id"]): IndexedItem(**row) for row in rows}
        self._items_by_video: dict[str, list[IndexedItem]] = defaultdict(list)
        for item in self._items.values():
            self._items_by_video[item.video_id].append(item)
        for items in self._items_by_video.values():
            items.sort(key=lambda item: (item.timestamp_start, item.item_id))

    def _to_chunk(self, item: IndexedItem, score: float | None = None) -> Chunk:
        return Chunk(
            item_id=item.item_id,
            video_id=item.video_id,
            modality=item.modality,
            timestamp_start=item.timestamp_start,
            timestamp_end=item.timestamp_end,
            text=item.text,
            frame_path=item.frame_path,
            source_id=item.source_id,
            score=score,
        )

    @staticmethod
    def _passes_filters(
        item: IndexedItem,
        *,
        modality: Modality | None = None,
        video_id: str | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> bool:
        if modality is not None and item.modality != modality:
            return False
        if video_id is not None and item.video_id != video_id:
            return False
        if t_start is not None and item.timestamp_end < t_start:
            return False
        if t_end is not None and item.timestamp_start > t_end:
            return False
        return True

    def semantic_search(
        self,
        query: str,
        k: int = 8,
        modality: Modality | None = None,
        video_id: str | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> list[Chunk]:
        query_embedding = self.embedder.embed_texts([query])
        raw_hits = self.vector_store.search(query_embedding, top_k=max(len(self._items), k), filters=None)
        matches: list[Chunk] = []
        for hit in raw_hits:
            item = self._items.get(hit.item_id)
            if item is None:
                continue
            if not self._passes_filters(
                item,
                modality=modality,
                video_id=video_id,
                t_start=t_start,
                t_end=t_end,
            ):
                continue
            matches.append(self._to_chunk(item, score=hit.score))
            if len(matches) >= k:
                break
        return matches

    def get_chunks_by_timestamp(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        modality: Modality | None = None,
    ) -> list[Chunk]:
        items = self._items_by_video.get(video_id, [])
        return [
            self._to_chunk(item)
            for item in items
            if self._passes_filters(item, modality=modality, video_id=video_id, t_start=t_start, t_end=t_end)
        ]

    def get_nearby_chunks(self, chunk_id: str, radius_seconds: float = 10.0) -> list[Chunk]:
        anchor = self._items.get(chunk_id)
        if anchor is None:
            return []
        lo = anchor.timestamp_start - radius_seconds
        hi = anchor.timestamp_end + radius_seconds
        items = self._items_by_video.get(anchor.video_id, [])
        return [
            self._to_chunk(item)
            for item in items
            if _interval_overlaps(item.timestamp_start, item.timestamp_end, lo, hi)
        ]

    def get_video_metadata(self, video_id: str) -> dict[str, Any]:
        items = self._items_by_video.get(video_id, [])
        if not items:
            return {}
        durations = [item.timestamp_end for item in items]
        counts_by_modality: dict[str, int] = defaultdict(int)
        for item in items:
            counts_by_modality[item.modality] += 1
        return {
            "video_id": video_id,
            "duration_sec": max(durations),
            "modalities": sorted(counts_by_modality.keys()),
            "counts_by_modality": dict(counts_by_modality),
            "num_indexed_items": len(items),
        }

    def resolve_chunks(self, item_ids: list[str]) -> list[Chunk]:
        out: list[Chunk] = []
        for item_id in item_ids:
            item = self._items.get(item_id)
            if item is not None:
                out.append(self._to_chunk(item))
        return out
