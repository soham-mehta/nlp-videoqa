from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

Modality = Literal["text", "frame"]


@dataclass(frozen=True)
class Chunk:
    item_id: str
    video_id: str
    modality: Modality
    timestamp_start: float
    timestamp_end: float
    text: str | None = None
    frame_path: str | None = None
    source_id: str | None = None
    score: Optional[float] = None


class RetrievalBackend(Protocol):
    def semantic_search(
        self,
        query: str,
        k: int = 8,
        modality: Optional[Modality] = None,
        video_id: Optional[str] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> list[Chunk]: ...

    def get_chunks_by_timestamp(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        modality: Optional[Modality] = None,
    ) -> list[Chunk]: ...

    def get_nearby_chunks(
        self,
        chunk_id: str,
        radius_seconds: float = 10.0,
    ) -> list[Chunk]: ...

    def get_video_metadata(self, video_id: str) -> dict: ...
