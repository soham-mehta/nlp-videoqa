from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    PerQueryMetrics,
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


def _process_item(
    rag_service: Any,
    item: Any,
    top_k_total: int,
    max_text_items: int,
    max_frame_items: int,
) -> tuple[Any, Any, Any]:
    """Run retrieval + generation for one benchmark item. Thread-safe."""
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
    elapsed = time.perf_counter() - t0
    return run, elapsed, item


def _build_question_result(
    item: Any,
    run: Any,
    elapsed: float,
    system_name: str,
    run_id: str,
    top_k_total: int,
    max_text_items: int,
    max_frame_items: int,
) -> tuple[BenchmarkQuestionResult, dict[str, Any], RetrievalMetrics, AnswerMetrics]:
    r_metrics = retrieval_metrics(run, item)
    a_metrics = answer_metrics(run.final_answer, item.gold_answer)

    gen_meta = run.generated_answer.metadata
    prompt_tok = gen_meta.get("prompt_tokens") or gen_meta.get("total_prompt_tokens") or 0
    compl_tok = gen_meta.get("completion_tokens") or gen_meta.get("total_completion_tokens") or 0
    num_llm = gen_meta.get("num_llm_calls", 1)
    num_tool = gen_meta.get("agent_total_tool_calls", 0)
    num_text = sum(1 for x in run.retrieved_items if x.modality == "text")
    num_frame = sum(1 for x in run.retrieved_items if x.modality == "frame")
    num_frames_sent = gen_meta.get("num_frames_sent", num_frame)

    pq_metrics = PerQueryMetrics(
        video_id=item.video_id,
        question_type=item.question_type,
        latency_sec=round(elapsed, 3),
        prompt_tokens=prompt_tok,
        completion_tokens=compl_tok,
        total_tokens=prompt_tok + compl_tok,
        num_llm_calls=num_llm,
        num_tool_calls=num_tool,
        num_retrieved_text=num_text,
        num_retrieved_frame=num_frame,
        num_frames_sent=num_frames_sent,
    )
    question_result = BenchmarkQuestionResult(
        question_id=item.question_id,
        retrieval_metrics=r_metrics,
        answer_metrics=a_metrics,
        final_answer=run.final_answer,
        per_query_metrics=pq_metrics,
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
    prediction_row = build_prediction_row(
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
    return question_result, prediction_row, r_metrics, a_metrics


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
    max_concurrent: int = 1,
) -> BenchmarkRunResult:
    items = load_benchmark_items(benchmark_path)
    run_id = str(uuid.uuid4())
    total = len(items)
    append_benchmark_progress(
        progress_file,
        f"[init] run_id={run_id} n_questions={total} benchmark={benchmark_path} concurrency={max_concurrent}",
    )

    progress_lock = threading.Lock()
    completed_count = 0
    pbar = tqdm(
        total=total,
        desc="Benchmark",
        unit="question",
        disable=not show_progress,
        leave=True,
        dynamic_ncols=True,
    )

    # Slots indexed by original position to preserve ordering.
    results_by_idx: dict[int, tuple[BenchmarkQuestionResult, dict[str, Any], RetrievalMetrics, AnswerMetrics]] = {}

    def _on_done(idx: int, item: Any, run: Any, elapsed: float) -> None:
        nonlocal completed_count
        qr, pr, rm, am = _build_question_result(
            item, run, elapsed, system_name, run_id,
            top_k_total, max_text_items, max_frame_items,
        )
        with progress_lock:
            results_by_idx[idx] = (qr, pr, rm, am)
            completed_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"{item.question_id[:30]} ({elapsed:.1f}s)", refresh=False)
        append_benchmark_progress(
            progress_file,
            f"[{completed_count}/{total}] DONE question_id={item.question_id} elapsed_s={elapsed:.1f}",
        )

    if max_concurrent <= 1:
        for idx, item in enumerate(items):
            append_benchmark_progress(
                progress_file,
                f"[{idx+1}/{total}] START question_id={item.question_id} video_id={item.video_id}",
            )
            run, elapsed, _ = _process_item(
                rag_service, item, top_k_total, max_text_items, max_frame_items,
            )
            _on_done(idx, item, run, elapsed)
    else:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {}
            for idx, item in enumerate(items):
                append_benchmark_progress(
                    progress_file,
                    f"[{idx+1}/{total}] SUBMIT question_id={item.question_id} video_id={item.video_id}",
                )
                fut = executor.submit(
                    _process_item,
                    rag_service, item, top_k_total, max_text_items, max_frame_items,
                )
                futures[fut] = idx
            for fut in as_completed(futures):
                idx = futures[fut]
                run, elapsed, item = fut.result()
                _on_done(idx, item, run, elapsed)

    pbar.close()
    append_benchmark_progress(
        progress_file,
        f"[loop_complete] run_id={run_id} all_{total}_questions_answered",
    )

    # Reassemble in original order.
    per_question: list[BenchmarkQuestionResult] = []
    prediction_rows: list[dict[str, Any]] = []
    retrieval_scores: list[RetrievalMetrics] = []
    answer_scores: list[AnswerMetrics] = []
    for idx in range(total):
        qr, pr, rm, am = results_by_idx[idx]
        per_question.append(qr)
        prediction_rows.append(pr)
        retrieval_scores.append(rm)
        answer_scores.append(am)

    pq_list = [q.per_query_metrics for q in per_question if q.per_query_metrics]
    aggregate_efficiency = {}
    if pq_list:
        aggregate_efficiency = {
            "mean_latency_sec": round(mean(m.latency_sec for m in pq_list), 3),
            "mean_prompt_tokens": round(mean(m.prompt_tokens for m in pq_list), 1),
            "mean_completion_tokens": round(mean(m.completion_tokens for m in pq_list), 1),
            "mean_total_tokens": round(mean(m.total_tokens for m in pq_list), 1),
            "total_prompt_tokens": sum(m.prompt_tokens for m in pq_list),
            "total_completion_tokens": sum(m.completion_tokens for m in pq_list),
            "total_tokens": sum(m.total_tokens for m in pq_list),
            "mean_llm_calls": round(mean(m.num_llm_calls for m in pq_list), 2),
            "mean_tool_calls": round(mean(m.num_tool_calls for m in pq_list), 2),
            "mean_frames_sent": round(mean(m.num_frames_sent for m in pq_list), 2),
        }
        qtypes = sorted(set(m.question_type for m in pq_list))
        per_type: dict[str, dict[str, float]] = {}
        for qt in qtypes:
            qt_a = [a for a, m in zip(answer_scores, pq_list) if m.question_type == qt]
            per_type[qt] = {
                "count": len(qt_a),
                "token_f1": round(mean(a.token_f1 for a in qt_a), 3) if qt_a else 0.0,
                "exact_match": round(mean(a.exact_match for a in qt_a), 3) if qt_a else 0.0,
            }
        aggregate_efficiency["per_question_type"] = per_type

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
            "aggregate_efficiency": aggregate_efficiency,
        },
    )
