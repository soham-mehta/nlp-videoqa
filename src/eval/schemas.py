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
class BenchmarkQuestionResult:
    question_id: str
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    final_answer: str
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRunResult:
    aggregate_retrieval_metrics: RetrievalMetrics
    aggregate_answer_metrics: AnswerMetrics
    per_question: list[BenchmarkQuestionResult]
    metadata: dict[str, Any] = field(default_factory=dict)
