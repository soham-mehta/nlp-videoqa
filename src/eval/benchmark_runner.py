from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

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


def append_benchmark_progress(path: str | None, line: str) -> None:
    """Append one UTF-8 line (no carriage returns) for tail -f while a run is in the background."""
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {line}\n")
        f.flush()


def run_benchmark(
    rag_service: BaselineRAGAnsweringService,
    benchmark_path: str,
    *,
    top_k_total: int = 8,
    max_text_items: int = 4,
    max_frame_items: int = 4,
    system_name: str = "baseline_rag",
    show_progress: bool = True,
    progress_file: str | None = None,
) -> BenchmarkRunResult:
    items = load_benchmark_items(benchmark_path)
    run_id = str(uuid.uuid4())
    total = len(items)
    append_benchmark_progress(
        progress_file,
        f"[init] run_id={run_id} n_questions={total} benchmark={benchmark_path}",
    )
    per_question: list[BenchmarkQuestionResult] = []
    prediction_rows: list[dict[str, Any]] = []
    retrieval_scores: list[RetrievalMetrics] = []
    answer_scores: list[AnswerMetrics] = []

    iterator = tqdm(
        items,
        desc="Benchmark",
        unit="question",
        total=total,
        disable=not show_progress,
        leave=True,
        dynamic_ncols=True,
    )
    for idx, item in enumerate(iterator, start=1):
        if show_progress:
            iterator.set_postfix_str(item.question_id[:40], refresh=False)
        append_benchmark_progress(
            progress_file,
            f"[{idx}/{total}] START question_id={item.question_id} video_id={item.video_id}",
        )
        t0 = time.perf_counter()
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
        elapsed = time.perf_counter() - t0
        append_benchmark_progress(
            progress_file,
            f"[{idx}/{total}] DONE question_id={item.question_id} elapsed_s={elapsed:.1f}",
        )

    append_benchmark_progress(
        progress_file,
        f"[loop_complete] run_id={run_id} all_{total}_questions_answered",
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
