from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.grading import grade_predictions
from src.utils.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade system predictions against benchmark ground truth.")
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--system-name", type=str, default="candidate_system")
    parser.add_argument("--output-json", type=Path, default=Path("data/eval/graded_report.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/eval/graded_per_question.jsonl"))
    args = parser.parse_args()

    report = grade_predictions(
        benchmark_path=str(args.benchmark_path),
        predictions_jsonl_path=str(args.predictions_jsonl),
        system_name=args.system_name,
    )
    write_json(args.output_json, report)
    write_jsonl(args.output_jsonl, report.get("per_question", []))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved graded report to: {args.output_json}")
    print(f"Saved per-question grades to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
