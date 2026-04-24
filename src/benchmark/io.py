from __future__ import annotations

from pathlib import Path
from typing import Any

from src.benchmark.schemas import BenchmarkItem, GoldEvidence
from src.utils.io import read_json, read_jsonl

_QUESTION_TYPE_MAP = {
    "factoid": "factoid",
    "procedure": "procedure",
    "temporal": "temporal",
    "visual": "visual",
    "multimodal": "multimodal",
    "other": "other",
    "factual": "factoid",
    "sequential": "procedure",
    "cross_modal": "multimodal",
    "summary": "other",
}

_MODALITY_MAP = {
    "text": "text",
    "frame": "frame",
    "image": "frame",
    "mixed": "mixed",
}


def _normalize_question_type(raw_question_type: Any) -> str:
    raw = str(raw_question_type)
    return _QUESTION_TYPE_MAP.get(raw, "other")


def _normalize_source_ids(gold_row: dict[str, Any]) -> list[str]:
    source_ids = gold_row.get("source_ids")
    if source_ids:
        return [str(x) for x in source_ids]
    merged: list[str] = []
    merged.extend(str(x) for x in gold_row.get("transcript_chunk_ids", []))
    merged.extend(str(x) for x in gold_row.get("frame_ids", []))
    return merged


def _normalize_gold_evidence(gold_row: dict[str, Any], *, video_id: str) -> GoldEvidence:
    raw_modality = str(gold_row["modality"])
    modality = _MODALITY_MAP.get(raw_modality, raw_modality)
    return GoldEvidence(
        video_id=str(gold_row.get("video_id", video_id)),
        modality=str(modality),  # type: ignore[arg-type]
        timestamp_start=float(gold_row["timestamp_start"]),
        timestamp_end=float(gold_row["timestamp_end"]),
        source_ids=_normalize_source_ids(gold_row),
    )


def _load_v1_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _load_v2_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    for video in payload:
        video_id = str(video["video_id"])
        for item in video.get("items", []):
            rows.append(
                {
                    "question_id": item["question_id"],
                    "video_id": video_id,
                    "question": item["question"],
                    "gold_answer": item.get("ideal_answer", ""),
                    "question_type": _normalize_question_type(item.get("question_type", "other")),
                    "question_type_raw": str(item.get("question_type", "")),
                    "gold_evidence": [
                        _normalize_gold_evidence(gold_row, video_id=video_id)
                        for gold_row in item.get("gold_evidence", [])
                    ],
                }
            )
    return rows


def _load_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _load_v2_json_rows(path)
    return _load_v1_jsonl_rows(path)


def load_benchmark_items(path: str) -> list[BenchmarkItem]:
    rows = _load_benchmark_rows(Path(path))
    items: list[BenchmarkItem] = []
    for row in rows:
        gold_rows = row.get("gold_evidence", [])
        gold = [
            g
            if isinstance(g, GoldEvidence)
            else _normalize_gold_evidence(g, video_id=str(row["video_id"]))
            for g in gold_rows
        ]
        items.append(
            BenchmarkItem(
                question_id=str(row["question_id"]),
                video_id=str(row["video_id"]),
                question=str(row["question"]),
                gold_answer=str(row["gold_answer"]),
                question_type=str(row["question_type"]),  # type: ignore[arg-type]
                gold_evidence=gold,
                question_type_raw=str(row.get("question_type_raw") or row["question_type"]),
            )
        )
    return items
