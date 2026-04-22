from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Modality = Literal["text", "frame"]


@dataclass(frozen=True)
class RetrievedItem:
    item_id: str
    video_id: str
    modality: Modality
    score: float
    timestamp_start: float
    timestamp_end: float
    text: str | None = None
    frame_path: str | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    items: list[RetrievedItem]
