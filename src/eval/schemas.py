from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalMetrics:
    top_k_hit: float
    evidence_overlap_hit: float
    evidence_recall_proxy: float


@dataclass(frozen=True)
class AnswerMetrics:
    exact_match: float
    normalized_match: float
    token_f1: float


@dataclass(frozen=True)
class PerQueryMetrics:
    """Top-level per-query fields surfaced for research paper analysis."""
    video_id: str
    question_type: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_llm_calls: int
    num_tool_calls: int
    num_retrieved_text: int
    num_retrieved_frame: int
    num_frames_sent: int


@dataclass(frozen=True)
class BenchmarkQuestionResult:
    question_id: str
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    final_answer: str
    per_query_metrics: PerQueryMetrics | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRunResult:
    aggregate_retrieval_metrics: RetrievalMetrics
    aggregate_answer_metrics: AnswerMetrics
    per_question: list[BenchmarkQuestionResult]
    prediction_rows: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
