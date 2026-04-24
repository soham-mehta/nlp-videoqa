from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.grading import grade_predictions
from src.utils.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade system predictions against benchmark ground truth.")
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=Path("data/eval/predictions_v1.jsonl"),
        help="prediction_v1 JSONL (e.g. from run_benchmark.py --predictions-jsonl).",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow legacy prediction rows without schema_version=prediction_v1.",
    )
    parser.add_argument("--system-name", type=str, default="candidate_system")
    parser.add_argument("--output-json", type=Path, default=Path("data/eval/graded_report.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/eval/graded_per_question.jsonl"))
    args = parser.parse_args()

    report = grade_predictions(
        benchmark_path=str(args.benchmark_path),
        predictions_jsonl_path=str(args.predictions_jsonl),
        system_name=args.system_name,
        strict_schema=not args.no_strict,
    )
    write_json(args.output_json, report)
    write_jsonl(args.output_jsonl, report.get("per_question", []))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved graded report to: {args.output_json}")
    print(f"Saved per-question grades to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
