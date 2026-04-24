from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QuestionType = Literal["factoid", "procedure", "temporal", "visual", "multimodal", "other"]


@dataclass(frozen=True)
class GoldEvidence:
    video_id: str
    modality: Literal["text", "frame", "mixed"]
    timestamp_start: float
    timestamp_end: float
    source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkItem:
    question_id: str
    video_id: str
    question: str
    gold_answer: str
    question_type: QuestionType
    gold_evidence: list[GoldEvidence] = field(default_factory=list)
    question_type_raw: str | None = None
