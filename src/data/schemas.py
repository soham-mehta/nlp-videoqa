from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Modality = Literal["text", "frame"]


@dataclass(frozen=True)
class TranscriptChunk:
    video_id: str
    source_id: str
    timestamp_start: float
    timestamp_end: float
    text: str


@dataclass(frozen=True)
class FrameItem:
    video_id: str
    source_id: str
    timestamp_start: float
    timestamp_end: float
    frame_path: str


@dataclass(frozen=True)
class IndexedItem:
    item_id: str
    video_id: str
    modality: Modality
    timestamp_start: float
    timestamp_end: float
    source_id: str
    text: str | None = None
    frame_path: str | None = None
