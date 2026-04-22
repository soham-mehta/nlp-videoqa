from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.benchmark.io import load_benchmark_items
from src.eval.metrics import answer_metrics
from src.eval.schemas import AnswerMetrics, RetrievalMetrics
from src.utils.io import read_jsonl


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return (a0 < b1) and (b0 < a1)


def _retrieval_metrics_from_prediction(
    benchmark_gold: list[dict[str, Any]],
    predicted_items: list[dict[str, Any]],
) -> RetrievalMetrics:
    if not benchmark_gold:
        return RetrievalMetrics(0.0, 0.0, 0.0)

    hit_flags: list[float] = []
    for gold in benchmark_gold:
        hit = 0.0
        for item in predicted_items:
            if str(item.get("video_id", "")) != str(gold["video_id"]):
                continue
            gold_mod = str(gold["modality"])
            if gold_mod != "mixed" and str(item.get("modality", "")) != gold_mod:
                continue
            if _interval_overlap(
                float(item.get("timestamp_start", 0.0)),
                float(item.get("timestamp_end", item.get("timestamp_start", 0.0))),
                float(gold["timestamp_start"]),
                float(gold["timestamp_end"]),
            ):
                hit = 1.0
                break
        hit_flags.append(hit)

    recall_proxy = sum(hit_flags) / len(hit_flags)
    return RetrievalMetrics(
        top_k_hit=float(any(x > 0 for x in hit_flags)),
        evidence_overlap_hit=float(any(x > 0 for x in hit_flags)),
        evidence_recall_proxy=recall_proxy,
    )


def grade_predictions(
    benchmark_path: str,
    predictions_jsonl_path: str,
    *,
    system_name: str = "candidate_system",
) -> dict[str, Any]:
    benchmark = load_benchmark_items(benchmark_path)
    benchmark_by_qid = {b.question_id: b for b in benchmark}
    preds = read_jsonl(Path(predictions_jsonl_path))

    per_question: list[dict[str, Any]] = []
    retrieval_scores: list[RetrievalMetrics] = []
    answer_scores: list[AnswerMetrics] = []

    for pred in preds:
        qid = str(pred.get("question_id", ""))
        if qid not in benchmark_by_qid:
            continue
        gold = benchmark_by_qid[qid]
        predicted_answer = str(pred.get("final_answer", pred.get("answer", "")))
        predicted_items = list(pred.get("retrieved_items", []))
        if not predicted_items:
            predicted_items = list(pred.get("debug_info", {}).get("retrieved_items", []))
        r = _retrieval_metrics_from_prediction(
            benchmark_gold=[
                {
                    "video_id": g.video_id,
                    "modality": g.modality,
                    "timestamp_start": g.timestamp_start,
                    "timestamp_end": g.timestamp_end,
                }
                for g in gold.gold_evidence
            ],
            predicted_items=predicted_items,
        )
        a = answer_metrics(predicted_answer, gold.gold_answer)
        retrieval_scores.append(r)
        answer_scores.append(a)
        per_question.append(
            {
                "question_id": qid,
                "question": gold.question,
                "gold_answer": gold.gold_answer,
                "predicted_answer": predicted_answer,
                "retrieval_metrics": asdict(r),
                "answer_metrics": asdict(a),
            }
        )

    def _avg(xs: list[float]) -> float:
        return mean(xs) if xs else 0.0

    return {
        "system_name": system_name,
        "benchmark_path": benchmark_path,
        "predictions_jsonl_path": predictions_jsonl_path,
        "num_questions_scored": len(per_question),
        "aggregate_retrieval_metrics": {
            "top_k_hit": _avg([x.top_k_hit for x in retrieval_scores]),
            "evidence_overlap_hit": _avg([x.evidence_overlap_hit for x in retrieval_scores]),
            "evidence_recall_proxy": _avg([x.evidence_recall_proxy for x in retrieval_scores]),
        },
        "aggregate_answer_metrics": {
            "exact_match": _avg([x.exact_match for x in answer_scores]),
            "normalized_match": _avg([x.normalized_match for x in answer_scores]),
            "token_f1": _avg([x.token_f1 for x in answer_scores]),
        },
        "per_question": per_question,
    }
