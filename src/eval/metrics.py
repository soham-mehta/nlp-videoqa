from __future__ import annotations

import re
from collections import Counter

from src.benchmark.schemas import BenchmarkItem, GoldEvidence
from src.eval.schemas import AnswerMetrics, RetrievalMetrics
from src.rag.schemas import RAGRunResult


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts.keys())
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(gold_tokens), 1)
    return (2 * precision * recall) / max(precision + recall, 1e-12)


def answer_metrics(predicted: str, gold: str) -> AnswerMetrics:
    return AnswerMetrics(
        exact_match=float(predicted.strip() == gold.strip()),
        normalized_match=float(_normalize_text(predicted) == _normalize_text(gold)),
        token_f1=_token_f1(predicted, gold),
    )


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return (a0 < b1) and (b0 < a1)


def _normalize_modality(modality: str) -> str:
    if modality == "image":
        return "frame"
    return modality


def _gold_hit(run: RAGRunResult, gold: GoldEvidence) -> bool:
    for item in run.retrieved_items:
        if item.video_id != gold.video_id:
            continue
        gold_modality = _normalize_modality(gold.modality)
        item_modality = _normalize_modality(item.modality)
        if gold_modality != "mixed" and item_modality != gold_modality:
            continue
        if _interval_overlap(
            item.timestamp_start,
            item.timestamp_end,
            gold.timestamp_start,
            gold.timestamp_end,
        ):
            return True
    return False


def retrieval_metrics(run: RAGRunResult, benchmark_item: BenchmarkItem) -> RetrievalMetrics:
    gold_list = benchmark_item.gold_evidence
    if not gold_list:
        return RetrievalMetrics(top_k_hit=0.0, evidence_overlap_hit=0.0, evidence_recall_proxy=0.0)

    hits = [1.0 if _gold_hit(run, gold) else 0.0 for gold in gold_list]
    recall_proxy = sum(hits) / len(hits)
    return RetrievalMetrics(
        top_k_hit=float(any(h > 0 for h in hits)),
        evidence_overlap_hit=float(any(h > 0 for h in hits)),
        evidence_recall_proxy=recall_proxy,
    )
