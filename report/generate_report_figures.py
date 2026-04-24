
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_main_results_table(
    baseline_run: dict,
    agentic_run: dict,
    judge_report: dict,
    out_dir: Path,
) -> None:
    rows = [
        {
            "metric": "Judge win rate",
            "baseline": judge_report["win_rate_baseline"],
            "agentic": judge_report["win_rate_agentic"],
        },
        {
            "metric": "Judge mean score",
            "baseline": judge_report["mean_score_baseline"],
            "agentic": judge_report["mean_score_agentic"],
        },
        {
            "metric": "Token F1",
            "baseline": baseline_run["aggregate_answer_metrics"]["token_f1"],
            "agentic": agentic_run["aggregate_answer_metrics"]["token_f1"],
        },
        {
            "metric": "Top-k hit",
            "baseline": baseline_run["aggregate_retrieval_metrics"]["top_k_hit"],
            "agentic": agentic_run["aggregate_retrieval_metrics"]["top_k_hit"],
        },
        {
            "metric": "Evidence overlap hit",
            "baseline": baseline_run["aggregate_retrieval_metrics"]["evidence_overlap_hit"],
            "agentic": agentic_run["aggregate_retrieval_metrics"]["evidence_overlap_hit"],
        },
        {
            "metric": "Evidence recall proxy",
            "baseline": baseline_run["aggregate_retrieval_metrics"]["evidence_recall_proxy"],
            "agentic": agentic_run["aggregate_retrieval_metrics"]["evidence_recall_proxy"],
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table_main_results.csv", index=False)


def write_efficiency_table(
    baseline_run: dict,
    agentic_run: dict,
    out_dir: Path,
) -> None:
    b = baseline_run["metadata"]["aggregate_efficiency"]
    a = agentic_run["metadata"]["aggregate_efficiency"]
    rows = [
        {"metric": "Mean latency (s)", "baseline": b["mean_latency_sec"], "agentic": a["mean_latency_sec"]},
        {"metric": "Mean prompt tokens", "baseline": b["mean_prompt_tokens"], "agentic": a["mean_prompt_tokens"]},
        {"metric": "Mean completion tokens", "baseline": b["mean_completion_tokens"], "agentic": a["mean_completion_tokens"]},
        {"metric": "Mean total tokens", "baseline": b["mean_total_tokens"], "agentic": a["mean_total_tokens"]},
        {"metric": "Total tokens", "baseline": b["total_tokens"], "agentic": a["total_tokens"]},
        {"metric": "Mean LLM calls", "baseline": b["mean_llm_calls"], "agentic": a["mean_llm_calls"]},
        {"metric": "Mean tool calls", "baseline": b["mean_tool_calls"], "agentic": a["mean_tool_calls"]},
        {"metric": "Mean frames sent", "baseline": b["mean_frames_sent"], "agentic": a["mean_frames_sent"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table_efficiency.csv", index=False)


def plot_overall_judge_outcomes(judge_report: dict, out_dir: Path) -> None:
    labels = ["Baseline wins", "Agentic wins", "Ties"]
    values = [
        judge_report["wins"]["baseline"],
        judge_report["wins"]["agentic"],
        judge_report["wins"]["tie"],
    ]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Number of questions")
    plt.title("Pairwise LLM-judge outcomes (100 benchmark questions)")
    for i, value in enumerate(values):
        plt.text(i, value + 1, str(value), ha="center")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_overall_judge_outcomes.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_type_judge_scores(judge_report: dict, out_dir: Path) -> None:
    type_order = ["factual", "temporal", "sequential", "cross_modal", "summary"]
    pretty = {
        "factual": "Factual",
        "temporal": "Temporal",
        "sequential": "Sequential",
        "cross_modal": "Cross-modal",
        "summary": "Summary",
    }

    baseline_scores = [judge_report["per_question_type"][t]["mean_score_baseline"] for t in type_order]
    agentic_scores = [judge_report["per_question_type"][t]["mean_score_agentic"] for t in type_order]
    x = range(len(type_order))
    width = 0.38

    plt.figure(figsize=(8, 4.8))
    plt.bar([i - width / 2 for i in x], baseline_scores, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], agentic_scores, width=width, label="Agentic")
    plt.xticks(list(x), [pretty[t] for t in type_order], rotation=20, ha="right")
    plt.ylabel("Mean judge score")
    plt.ylim(0, 5.5)
    plt.title("Mean pairwise judge score by question type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_per_type_judge_scores.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_efficiency_comparison(
    baseline_run: dict,
    agentic_run: dict,
    out_dir: Path,
) -> None:
    b = baseline_run["metadata"]["aggregate_efficiency"]
    a = agentic_run["metadata"]["aggregate_efficiency"]

    metrics = ["Mean latency (s)", "Mean total tokens", "Mean LLM calls", "Mean tool calls"]
    baseline_vals = [b["mean_latency_sec"], b["mean_total_tokens"], b["mean_llm_calls"], b["mean_tool_calls"]]
    agentic_vals = [a["mean_latency_sec"], a["mean_total_tokens"], a["mean_llm_calls"], a["mean_tool_calls"]]

    x = range(len(metrics))
    width = 0.38

    plt.figure(figsize=(8.5, 4.8))
    plt.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], agentic_vals, width=width, label="Agentic")
    plt.xticks(list(x), metrics, rotation=20, ha="right")
    plt.ylabel("Value")
    plt.title("Efficiency comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_efficiency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_tool_usage(agentic_detail: pd.DataFrame, out_dir: Path) -> None:
    tool_counts: Dict[str, int] = {
        "semantic_search": 0,
        "get_chunks_by_timestamp": 0,
        "get_nearby_chunks": 0,
        "get_video_metadata": 0,
        "final_answer": 0,
    }

    for _, row in agentic_detail.iterrows():
        run_debug = row.get("debug_info", {}).get("run_debug", {})
        trajectory = run_debug.get("agent_trajectory", []) or []
        for step in trajectory:
            tool = step.get("tool")
            if tool in tool_counts:
                tool_counts[tool] += 1

    labels = list(tool_counts.keys())
    values = [tool_counts[k] for k in labels]

    plt.figure(figsize=(8, 4.6))
    plt.bar(labels, values)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Number of calls")
    plt.title("Tool usage in the agentic system")
    for i, value in enumerate(values):
        plt.text(i, value + max(values) * 0.02 if max(values) > 0 else 0.02, str(value), ha="center")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_tool_usage.png", dpi=300, bbox_inches="tight")
    plt.close()

    pd.DataFrame({"tool": labels, "count": values}).to_csv(out_dir / "table_tool_usage.csv", index=False)


def select_case_studies(
    judge_per_question: pd.DataFrame,
    baseline_detail: pd.DataFrame,
    agentic_detail: pd.DataFrame,
    out_dir: Path,
) -> None:
    merged = judge_per_question.copy()
    merged["score_margin"] = merged["score_baseline"] - merged["score_agentic"]

    best_baseline = merged.sort_values("score_margin", ascending=False).iloc[0]
    best_agentic = merged.sort_values("score_margin", ascending=True).iloc[0]

    baseline_map = baseline_detail.set_index("question_id").to_dict("index")
    agentic_map = agentic_detail.set_index("question_id").to_dict("index")

    selected_rows: List[dict] = []
    for label, row in [("baseline_case", best_baseline), ("agentic_case", best_agentic)]:
        qid = row["question_id"]
        b = baseline_map[qid]
        a = agentic_map[qid]
        selected_rows.append(
            {
                "case_label": label,
                "question_id": qid,
                "video_id": row["video_id"],
                "question_type": row["question_type"],
                "winner": row["winner"],
                "judge_score_baseline": row["score_baseline"],
                "judge_score_agentic": row["score_agentic"],
                "judge_reasoning": row["reasoning"],
                "question": b["debug_info"]["question"],
                "gold_answer": b["debug_info"]["gold_answer"],
                "baseline_answer": b["final_answer"],
                "agentic_answer": a["final_answer"],
            }
        )

    pd.DataFrame(selected_rows).to_csv(out_dir / "table_case_studies.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures and tables for the video QA project.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing the result files.")
    parser.add_argument("--output-dir", type=Path, default=Path("report_outputs"), help="Directory to save figures and tables.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    baseline_run = load_json(args.input_dir / "baseline_benchmark_run.json")
    agentic_run = load_json(args.input_dir / "agentic_benchmark_run.json")
    judge_report = load_json(args.input_dir / "llm_judge_report.json")
    baseline_detail = load_jsonl(args.input_dir / "baseline_detail.jsonl")
    agentic_detail = load_jsonl(args.input_dir / "agentic_detail.jsonl")
    judge_per_question = load_jsonl(args.input_dir / "llm_judge_per_question.jsonl")

    write_main_results_table(baseline_run, agentic_run, judge_report, args.output_dir)
    write_efficiency_table(baseline_run, agentic_run, args.output_dir)
    plot_overall_judge_outcomes(judge_report, args.output_dir)
    plot_per_type_judge_scores(judge_report, args.output_dir)
    plot_efficiency_comparison(baseline_run, agentic_run, args.output_dir)
    plot_tool_usage(agentic_detail, args.output_dir)
    select_case_studies(judge_per_question, baseline_detail, agentic_detail, args.output_dir)

    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
