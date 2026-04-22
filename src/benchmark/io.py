from __future__ import annotations

from pathlib import Path

from src.benchmark.schemas import BenchmarkItem, GoldEvidence
from src.utils.io import read_jsonl


def load_benchmark_items(path: str) -> list[BenchmarkItem]:
    rows = read_jsonl(Path(path))
    items: list[BenchmarkItem] = []
    for row in rows:
        gold = [
            GoldEvidence(
                video_id=str(g["video_id"]),
                modality=str(g["modality"]),  # type: ignore[arg-type]
                timestamp_start=float(g["timestamp_start"]),
                timestamp_end=float(g["timestamp_end"]),
                source_ids=[str(x) for x in g.get("source_ids", [])],
            )
            for g in row.get("gold_evidence", [])
        ]
        items.append(
            BenchmarkItem(
                question_id=str(row["question_id"]),
                video_id=str(row["video_id"]),
                question=str(row["question"]),
                gold_answer=str(row["gold_answer"]),
                question_type=str(row["question_type"]),  # type: ignore[arg-type]
                gold_evidence=gold,
            )
        )
    return items
