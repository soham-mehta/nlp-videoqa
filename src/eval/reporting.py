from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.utils.io import write_json, write_jsonl


def save_benchmark_run(
    output_json_path: Path | None,
    output_jsonl_path: Path | None,
    run_result: dict[str, Any],
    per_question_rows: list[dict[str, Any]],
) -> None:
    if output_json_path is not None:
        write_json(output_json_path, run_result)
    if output_jsonl_path is not None:
        write_jsonl(output_jsonl_path, per_question_rows)


def save_answer_run(output_json_path: Path, run_result: dict[str, Any]) -> None:
    write_json(output_json_path, run_result)


def dataclass_to_dict(obj: object) -> dict[str, Any]:
    return asdict(obj)
