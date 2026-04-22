from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retrieval.schemas import RetrievedItem


@dataclass(frozen=True)
class RetrievalPolicy:
    top_k_total: int = 8
    max_text_items: int = 4
    max_frame_items: int = 4
    dedupe_seconds: float = 1.0


@dataclass(frozen=True)
class AnswerRequest:
    question: str
    video_id: str | None = None
    system_prompt: str | None = None
    retrieval_policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)


@dataclass(frozen=True)
class TranscriptEvidence:
    evidence_id: str
    video_id: str
    timestamp_start: float
    timestamp_end: float
    text: str
    source_id: str | None = None


@dataclass(frozen=True)
class FrameEvidence:
    evidence_id: str
    video_id: str
    timestamp_start: float
    timestamp_end: float
    frame_path: str
    source_id: str | None = None
    caption: str | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    question: str
    transcripts: list[TranscriptEvidence]
    frames: list[FrameEvidence]
    retrieved_items: list[RetrievedItem]


@dataclass(frozen=True)
class GeneratedAnswer:
    answer_text: str
    model_name: str
    prompt_text: str
    used_evidence_ids: list[str]
    raw_response_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RAGRunResult:
    request: AnswerRequest
    final_answer: str
    retrieved_items: list[RetrievedItem]
    evidence_bundle: EvidenceBundle
    generated_answer: GeneratedAnswer
    debug_info: dict[str, Any] = field(default_factory=dict)
