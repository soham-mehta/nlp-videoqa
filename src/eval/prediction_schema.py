from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.retrieval.schemas import RetrievedItem

SCHEMA_VERSION = "prediction_v1"

Modality = Literal["text", "frame"]


@dataclass(frozen=True)
class RetrievedEvidenceItemV1:
    """One retrieved evidence item in the shared prediction format."""

    item_id: str
    video_id: str
    modality: Modality
    score: float
    timestamp_start: float
    timestamp_end: float
    text: str | None = None
    frame_path: str | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class PredictionRecordV1:
    """
    One line of predictions JSONL for cross-system benchmarking and grading.

    Required for graders: question_id, video_id, question, final_answer, retrieved_items.
    """

    schema_version: str
    system_name: str
    run_id: str
    question_id: str
    video_id: str
    question: str
    final_answer: str
    retrieved_items: list[RetrievedEvidenceItemV1]
    config: dict[str, Any] = field(default_factory=dict)


def retrieved_item_to_evidence_v1(item: RetrievedItem) -> RetrievedEvidenceItemV1:
    return RetrievedEvidenceItemV1(
        item_id=item.item_id,
        video_id=item.video_id,
        modality=item.modality,
        score=float(item.score),
        timestamp_start=float(item.timestamp_start),
        timestamp_end=float(item.timestamp_end),
        text=item.text,
        frame_path=item.frame_path,
        source_id=item.source_id,
    )


def build_prediction_row(
    *,
    system_name: str,
    run_id: str,
    question_id: str,
    video_id: str,
    question: str,
    final_answer: str,
    retrieved_items: list[RetrievedItem],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = PredictionRecordV1(
        schema_version=SCHEMA_VERSION,
        system_name=system_name,
        run_id=run_id,
        question_id=question_id,
        video_id=video_id,
        question=question,
        final_answer=final_answer,
        retrieved_items=[retrieved_item_to_evidence_v1(x) for x in retrieved_items],
        config=dict(config or {}),
    )
    return asdict(record)


def validate_prediction_row(row: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors; empty list means OK."""
    errors: list[str] = []
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}, got {row.get('schema_version')!r}")
    for key in ("system_name", "run_id", "question_id", "video_id", "question", "final_answer"):
        if key not in row or row[key] is None:
            errors.append(f"missing required field: {key}")
    if "retrieved_items" not in row or not isinstance(row["retrieved_items"], list):
        errors.append("retrieved_items must be a non-null list")
        return errors
    for i, it in enumerate(row["retrieved_items"]):
        if not isinstance(it, dict):
            errors.append(f"retrieved_items[{i}] must be an object")
            continue
        for k in ("item_id", "video_id", "modality", "score", "timestamp_start", "timestamp_end"):
            if k not in it:
                errors.append(f"retrieved_items[{i}] missing {k}")
        mod = it.get("modality")
        if mod not in ("text", "frame"):
            errors.append(f"retrieved_items[{i}].modality must be 'text' or 'frame', got {mod!r}")
    return errors
