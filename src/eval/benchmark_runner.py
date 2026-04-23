from __future__ import annotations

import uuid
from dataclasses import asdict
from statistics import mean
from typing import Any

from src.benchmark.io import load_benchmark_items
from src.eval.metrics import answer_metrics, retrieval_metrics
from src.eval.prediction_schema import build_prediction_row
from src.eval.schemas import (
    AnswerMetrics,
    BenchmarkQuestionResult,
    BenchmarkRunResult,
    RetrievalMetrics,
)
from src.rag.schemas import AnswerRequest
from src.rag.answering import BaselineRAGAnsweringService
from src.rag.schemas import RetrievalPolicy


def _avg_retrieval(xs: list[RetrievalMetrics]) -> RetrievalMetrics:
    if not xs:
        return RetrievalMetrics(0.0, 0.0, 0.0)
    return RetrievalMetrics(
        top_k_hit=mean([x.top_k_hit for x in xs]),
        evidence_overlap_hit=mean([x.evidence_overlap_hit for x in xs]),
        evidence_recall_proxy=mean([x.evidence_recall_proxy for x in xs]),
    )


def _avg_answer(xs: list[AnswerMetrics]) -> AnswerMetrics:
    if not xs:
        return AnswerMetrics(0.0, 0.0, 0.0)
    return AnswerMetrics(
        exact_match=mean([x.exact_match for x in xs]),
        normalized_match=mean([x.normalized_match for x in xs]),
        token_f1=mean([x.token_f1 for x in xs]),
    )


def run_benchmark(
    rag_service: BaselineRAGAnsweringService,
    benchmark_path: str,
    *,
    top_k_total: int = 8,
    max_text_items: int = 4,
    max_frame_items: int = 4,
    system_name: str = "baseline_rag",
) -> BenchmarkRunResult:
    items = load_benchmark_items(benchmark_path)
    run_id = str(uuid.uuid4())
    per_question: list[BenchmarkQuestionResult] = []
    prediction_rows: list[dict[str, Any]] = []
    retrieval_scores: list[RetrievalMetrics] = []
    answer_scores: list[AnswerMetrics] = []

    for item in items:
        request = AnswerRequest(
            question=item.question,
            video_id=item.video_id,
            retrieval_policy=RetrievalPolicy(
                top_k_total=top_k_total,
                max_text_items=max_text_items,
                max_frame_items=max_frame_items,
            ),
        )
        run = rag_service.run(request)
        r_metrics = retrieval_metrics(run, item)
        a_metrics = answer_metrics(run.final_answer, item.gold_answer)
        retrieval_scores.append(r_metrics)
        answer_scores.append(a_metrics)
        per_question.append(
            BenchmarkQuestionResult(
                question_id=item.question_id,
                retrieval_metrics=r_metrics,
                answer_metrics=a_metrics,
                final_answer=run.final_answer,
                debug_info={
                    "question": item.question,
                    "gold_answer": item.gold_answer,
                    "retrieved_items": [asdict(x) for x in run.retrieved_items],
                    "evidence_bundle": asdict(run.evidence_bundle),
                    "prompt_text": run.generated_answer.prompt_text,
                    "generator_metadata": run.generated_answer.metadata,
                    "run_debug": run.debug_info,
                },
            )
        )
        prediction_rows.append(
            build_prediction_row(
                system_name=system_name,
                run_id=run_id,
                question_id=item.question_id,
                video_id=item.video_id,
                question=item.question,
                final_answer=run.final_answer,
                retrieved_items=run.retrieved_items,
                config={
                    "top_k_total": top_k_total,
                    "max_text_items": max_text_items,
                    "max_frame_items": max_frame_items,
                },
            )
        )

    return BenchmarkRunResult(
        aggregate_retrieval_metrics=_avg_retrieval(retrieval_scores),
        aggregate_answer_metrics=_avg_answer(answer_scores),
        per_question=per_question,
        prediction_rows=prediction_rows,
        metadata={
            "benchmark_path": benchmark_path,
            "num_questions": len(items),
            "system_name": system_name,
            "run_id": run_id,
        },
    )
