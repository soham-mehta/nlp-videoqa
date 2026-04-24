
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def aggregate_efficiency(run: dict) -> dict:
    rows = []
    for item in run.get("per_question", []):
        pq = item.get("per_query_metrics", {})
        rows.append(
            {
                "latency_sec": pq.get("latency_sec", 0.0),
                "total_tokens": pq.get("total_tokens", 0.0),
                "num_llm_calls": pq.get("num_llm_calls", 0.0),
                "num_tool_calls": pq.get("num_tool_calls", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "mean_latency_sec": 0.0,
            "mean_total_tokens": 0.0,
            "mean_llm_calls": 0.0,
            "mean_tool_calls": 0.0,
        }
    return {
        "mean_latency_sec": df["latency_sec"].mean(),
        "mean_total_tokens": df["total_tokens"].mean(),
        "mean_llm_calls": df["num_llm_calls"].mean(),
        "mean_tool_calls": df["num_tool_calls"].mean(),
    }


def tool_usage_total(run: dict) -> int:
    total = 0
    for item in run.get("per_question", []):
        for step in item.get("debug_info", {}).get("run_debug", {}).get("agent_trajectory", []) or []:
            if step.get("tool"):
                total += 1
    return total


def create_per_type_heatmap(
    baseline_judge_report: dict,
    judge_reports: Dict[str, dict],
    out_dir: Path,
) -> None:
    type_order = ["factual", "temporal", "sequential", "cross_modal", "summary"]
    systems = ["Baseline"] + list(judge_reports.keys())

    data = []
    for qtype in type_order:
        row = [baseline_judge_report["per_question_type"][qtype]["mean_score_baseline"]]
        for label, report in judge_reports.items():
            row.append(report["per_question_type"][qtype]["mean_score_agentic"])
        data.append(row)

    df = pd.DataFrame(
        data,
        index=["Factual", "Temporal", "Sequential", "Cross-modal", "Summary"],
        columns=systems,
    )
    df.to_csv(out_dir / "table_nonbar_per_type_heatmap.csv")

    plt.figure(figsize=(8.0, 4.8))
    plt.imshow(df.values, aspect="auto")
    plt.xticks(range(len(df.columns)), df.columns, rotation=20, ha="right")
    plt.yticks(range(len(df.index)), df.index)
    plt.colorbar(label="Mean judge score")
    plt.title("Question type × system judge-score heatmap")
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            plt.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_nonbar_per_type_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_metric_dumbbell(
    baseline_run: dict,
    best_agentic_run: dict,
    best_agentic_label: str,
    best_judge_report: dict,
    out_dir: Path,
) -> None:
    metrics = [
        ("Judge win rate", best_judge_report["win_rate_baseline"], best_judge_report["win_rate_agentic"]),
        ("Mean judge score", best_judge_report["mean_score_baseline"], best_judge_report["mean_score_agentic"]),
        ("Token F1", baseline_run["aggregate_answer_metrics"]["token_f1"], best_agentic_run["aggregate_answer_metrics"]["token_f1"]),
        ("Top-k hit", baseline_run["aggregate_retrieval_metrics"]["top_k_hit"], best_agentic_run["aggregate_retrieval_metrics"]["top_k_hit"]),
        ("Overlap hit", baseline_run["aggregate_retrieval_metrics"]["evidence_overlap_hit"], best_agentic_run["aggregate_retrieval_metrics"]["evidence_overlap_hit"]),
        ("Recall proxy", baseline_run["aggregate_retrieval_metrics"]["evidence_recall_proxy"], best_agentic_run["aggregate_retrieval_metrics"]["evidence_recall_proxy"]),
    ]
    y = list(range(len(metrics)))

    plt.figure(figsize=(8.2, 5.0))
    for i, (name, b, a) in enumerate(metrics):
        plt.plot([b, a], [i, i], linewidth=2)
        plt.scatter([b, a], [i, i], s=50)
        plt.text(b - 0.01, i + 0.12, f"{b:.2f}", ha="right", fontsize=9)
        plt.text(a + 0.01, i + 0.12, f"{a:.2f}", ha="left", fontsize=9)
    plt.yticks(y, [m[0] for m in metrics])
    plt.xlabel("Score")
    plt.title(f"Baseline vs {best_agentic_label}: metric dumbbell plot")
    plt.xlim(0, max(max(m[1], m[2]) for m in metrics) + 0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_nonbar_metric_dumbbell.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_judge_score_boxplot(
    baseline_per_q: pd.DataFrame,
    per_q_frames: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    data = []
    labels = []

    data.append(baseline_per_q["score_baseline"].tolist())
    labels.append("Baseline")

    for label, df in per_q_frames.items():
        data.append(df["score_agentic"].tolist())
        labels.append(label)

    # Cohesive palette: neutral baseline, then cooler hues for agentic sizes.
    facecolors = ["#7d8692", "#6ea8d9", "#3da89b", "#2c5364"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor="white")
    ax.set_facecolor("#f6f7f9")

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#c45c3e", "linewidth": 2.0},
        whiskerprops={"color": "#3d3d3d", "linewidth": 1.15, "linestyle": "-"},
        capprops={"color": "#3d3d3d", "linewidth": 1.15},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#c45c3e",
            "markeredgecolor": "none",
            "markersize": 5,
            "alpha": 0.45,
        },
        boxprops={"linewidth": 1.1, "edgecolor": "#2d3436"},
    )
    for patch, fc in zip(bp["boxes"], facecolors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#2d3436")

    ax.set_ylabel("Per-question judge score", fontsize=11)
    ax.set_title("Distribution of judge scores by system", fontsize=12, fontweight="semibold", pad=10)
    ax.set_ylim(-0.2, 5.35)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.55, color="#b8bec6")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c5cad3")
    ax.spines["bottom"].set_color("#c5cad3")
    ax.tick_params(axis="both", labelsize=10, colors="#333")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_nonbar_judge_score_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_tool_performance_bubble(
    judge_reports: Dict[str, dict],
    agentic_runs: Dict[str, dict],
    out_dir: Path,
) -> None:
    rows = []
    for label, report in judge_reports.items():
        run = agentic_runs[label]
        eff = aggregate_efficiency(run)
        rows.append(
            {
                "model": label,
                "mean_judge_score": report["mean_score_agentic"],
                "mean_tool_calls": eff["mean_tool_calls"],
                "mean_total_tokens": eff["mean_total_tokens"],
                "total_tool_calls": tool_usage_total(run),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table_nonbar_tool_performance_bubble.csv", index=False)

    sizes = [max(80, x / 40.0) for x in df["mean_total_tokens"]]
    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(df["mean_tool_calls"], df["mean_judge_score"], s=sizes)
    for _, row in df.iterrows():
        plt.text(row["mean_tool_calls"] + 0.02, row["mean_judge_score"] + 0.02, row["model"], fontsize=9)
    plt.xlabel("Mean tool calls per question")
    plt.ylabel("Mean judge score")
    plt.title("Tool-use / performance tradeoff across agentic models")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_nonbar_tool_performance_bubble.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate non-bar plots for the report.")
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--judge-report-8b", type=Path, required=True)
    parser.add_argument("--judge-report-30b", type=Path, required=True)
    parser.add_argument("--judge-report-32b", type=Path, required=True)
    parser.add_argument("--judge-per-question-8b", type=Path, required=True)
    parser.add_argument("--judge-per-question-30b", type=Path, required=True)
    parser.add_argument("--judge-per-question-32b", type=Path, required=True)
    parser.add_argument("--agentic-run-8b", type=Path, required=True)
    parser.add_argument("--agentic-run-30b", type=Path, required=True)
    parser.add_argument("--agentic-run-32b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("report_nonbar_outputs"))
    parser.add_argument("--best-agentic-label", type=str, default="30b")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    baseline_run = load_json(args.baseline_run)

    judge_reports = {
        "8b": load_json(args.judge_report_8b),
        "30b": load_json(args.judge_report_30b),
        "32b": load_json(args.judge_report_32b),
    }

    per_q_frames = {
        "8b": load_jsonl(args.judge_per_question_8b),
        "30b": load_jsonl(args.judge_per_question_30b),
        "32b": load_jsonl(args.judge_per_question_32b),
    }

    agentic_runs = {
        "8b": load_json(args.agentic_run_8b),
        "30b": load_json(args.agentic_run_30b),
        "32b": load_json(args.agentic_run_32b),
    }

    # Use the 8b judge file to get baseline per-question scores; baseline side is shared within each comparison file.
    baseline_per_q = per_q_frames["8b"]

    create_per_type_heatmap(judge_reports["8b"], judge_reports, args.output_dir)
    create_metric_dumbbell(
        baseline_run,
        agentic_runs[args.best_agentic_label],
        args.best_agentic_label,
        judge_reports[args.best_agentic_label],
        args.output_dir,
    )
    create_judge_score_boxplot(baseline_per_q, per_q_frames, args.output_dir)
    create_tool_performance_bubble(judge_reports, agentic_runs, args.output_dir)

    print(f"Saved non-bar plots to {args.output_dir}")


if __name__ == "__main__":
    main()
