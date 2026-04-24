#!/usr/bin/env python3
"""Generate the seven priority paper plots from bundled report JSON (run1 / runb / runa)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_question_type(raw: str) -> str:
    m = {
        "factoid": "factual",
        "factual": "factual",
        "temporal": "temporal",
        "procedure": "sequential",
        "sequential": "sequential",
        "multimodal": "cross_modal",
        "cross_modal": "cross_modal",
        "other": "summary",
        "summary": "summary",
    }
    return m.get(raw or "other", "summary")


TYPE_ORDER = ["factual", "temporal", "sequential", "cross_modal", "summary"]
TYPE_LABELS = ["Factual", "Temporal", "Sequential", "Cross-modal", "Summary"]


def token_f1_by_type(run: dict) -> Dict[str, float]:
    sums: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for item in run.get("per_question", []):
        pq = item.get("per_query_metrics", {})
        qt = normalize_question_type(pq.get("question_type", "other"))
        f1 = float(item.get("answer_metrics", {}).get("token_f1", 0.0))
        sums[qt] += f1
        counts[qt] += 1
    out: Dict[str, float] = {}
    for t in TYPE_ORDER:
        if counts[t]:
            out[t] = sums[t] / counts[t]
        else:
            out[t] = float("nan")
    return out


def mean_per_question(run: dict, key: str) -> float:
    vals: List[float] = []
    for item in run.get("per_question", []):
        pq = item.get("per_query_metrics", {})
        if key in pq:
            vals.append(float(pq[key]))
    return float(np.mean(vals)) if vals else 0.0


def stop_reason_counts(run: dict) -> Counter[str]:
    c: Counter[str] = Counter()
    for item in run.get("per_question", []):
        gm = item.get("debug_info", {}).get("generator_metadata", {})
        rd = item.get("debug_info", {}).get("run_debug", {})
        r = gm.get("agent_stopped_reason") or rd.get("agent_stopped_reason") or "unknown"
        c[str(r)] += 1
    return c


def plot1_baseline_vs_agentic_f1(
    baseline_run: dict,
    agentic_by_label: Dict[str, dict],
    out_dir: Path,
) -> None:
    baseline_f1 = float(baseline_run["aggregate_answer_metrics"]["token_f1"])
    labels = list(agentic_by_label.keys())
    agentic_f1s = [float(agentic_by_label[k]["aggregate_answer_metrics"]["token_f1"]) for k in labels]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
    ax.set_facecolor("#f6f7f9")
    ax.bar(x - w / 2, [baseline_f1] * len(labels), width=w, label="Baseline", color="#7d8692", edgecolor="#2d3436", linewidth=0.8)
    ax.bar(x + w / 2, agentic_f1s, width=w, label="Agentic", color="#3da89b", edgecolor="#2d3436", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb} agent" for lb in labels])
    ax.set_ylabel("Aggregate token F1")
    ax.set_title("Baseline vs agentic token F1 (by model)")
    ax.set_ylim(0, max([baseline_f1, *agentic_f1s, 0.1]) * 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_01_baseline_vs_agentic_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot2_context_ablation_f1(run_a: dict, run_b: dict, label_a: str, label_b: str, out_dir: Path) -> None:
    f1_a = float(run_a["aggregate_answer_metrics"]["token_f1"])
    f1_b = float(run_b["aggregate_answer_metrics"]["token_f1"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor="white")
    ax.set_facecolor("#f6f7f9")
    colors = ["#c45c3e", "#2a9d8f"]
    ax.bar([0, 1], [f1_a, f1_b], color=colors, edgecolor="#2d3436", linewidth=0.8, width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b], rotation=12, ha="right")
    ax.set_ylabel("Aggregate token F1")
    ax.set_title("Context ablation: agentic token F1")
    ax.set_ylim(0, max(f1_a, f1_b, 0.05) * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_02_context_ablation_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot3_f1_heatmap_baseline_and_agentic(
    baseline_run: dict,
    agentic_by_label: Dict[str, dict],
    out_dir: Path,
) -> None:
    b_row = [token_f1_by_type(baseline_run).get(t, float("nan")) for t in TYPE_ORDER]
    rows = [b_row]
    for lb in agentic_by_label:
        rows.append([token_f1_by_type(agentic_by_label[lb]).get(t, float("nan")) for t in TYPE_ORDER])
    arr = np.array(rows, dtype=float)
    cols = ["Baseline"] + [f"{lb} agentic" for lb in agentic_by_label]

    fig, ax = plt.subplots(figsize=(9.0, 4.2), facecolor="white")
    im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1.0)
    ax.set_xticks(range(len(TYPE_LABELS)))
    ax.set_xticklabels(TYPE_LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Token F1 by question type (baseline vs agentic)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Token F1")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_03_f1_by_type_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot4_tool_calls_vs_f1(agentic_by_label: Dict[str, dict], out_dir: Path) -> None:
    xs, ys, labels, hits = [], [], [], []
    for lb, run in agentic_by_label.items():
        xs.append(mean_per_question(run, "num_tool_calls"))
        ys.append(float(run["aggregate_answer_metrics"]["token_f1"]))
        labels.append(lb)
        hits.append(float(run["aggregate_retrieval_metrics"]["top_k_hit"]))

    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor="white")
    ax.set_facecolor("#f6f7f9")
    ax.scatter(xs, ys, s=[220, 260, 240], c=["#6ea8d9", "#3da89b", "#2c5364"], edgecolors="#2d3436", linewidths=0.8, zorder=3)
    for x, y, lb, h in zip(xs, ys, labels, hits):
        ax.annotate(f"{lb}\n(top-k {h:.2f})", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Mean tool calls per question")
    ax.set_ylabel("Aggregate token F1")
    ax.set_title("Tool calls vs token F1 (with top-k hit in label)")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_04_tool_calls_vs_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot5_stop_reason_stacked(agentic_by_label: Dict[str, dict], out_dir: Path) -> None:
    all_reasons: set[str] = set()
    per_model: Dict[str, Counter[str]] = {}
    for lb, run in agentic_by_label.items():
        c = stop_reason_counts(run)
        per_model[lb] = c
        all_reasons.update(c.keys())
    order = sorted(all_reasons, key=lambda r: -sum(per_model[lb].get(r, 0) for lb in per_model))
    if "unknown" in order:
        order.remove("unknown")
        order.append("unknown")

    models = list(agentic_by_label.keys())
    fig, ax = plt.subplots(figsize=(7.8, 4.8), facecolor="white")
    ax.set_facecolor("#f6f7f9")
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    cmap = colormaps["tab10"]
    for i, reason in enumerate(order):
        heights = [per_model[m].get(reason, 0) for m in models]
        if sum(heights) == 0:
            continue
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=reason.replace("_", " "),
            color=cmap(i % 10),
            edgecolor="#2d3436",
            linewidth=0.4,
        )
        bottoms += np.array(heights, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m} agentic" for m in models])
    ax.set_ylabel("Questions (count)")
    ax.set_title("Agent stop reasons by model (runb agentic runs)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_05_stop_reason_stacked.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot6_judge_wins_horizontal(judge_reports: Dict[str, dict], out_dir: Path) -> None:
    order = ["8b", "30b", "32b"]
    rows = []
    for key in order:
        if key not in judge_reports:
            continue
        w = judge_reports[key]["wins"]
        rows.append(
            (
                f"{key} judge",
                int(w.get("baseline", 0)),
                int(w.get("tie", 0)),
                int(w.get("agentic", 0)),
                int(w.get("parse_error", 0)),
            )
        )
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8.0, 3.8), facecolor="white")
    ax.set_facecolor("#f6f7f9")
    y = np.arange(len(rows))
    labels_r = [r[0] for r in rows]
    b = np.array([r[1] for r in rows], dtype=float)
    t = np.array([r[2] for r in rows], dtype=float)
    a = np.array([r[3] for r in rows], dtype=float)
    pe = np.array([r[4] for r in rows], dtype=float)

    ax.barh(y, b, color="#7d8692", label="Baseline wins", edgecolor="#2d3436", linewidth=0.5)
    ax.barh(y, t, left=b, color="#b8b8b8", label="Ties", edgecolor="#2d3436", linewidth=0.5)
    ax.barh(y, a, left=b + t, color="#3da89b", label="Agentic wins", edgecolor="#2d3436", linewidth=0.5)
    if pe.sum() > 0:
        ax.barh(y, pe, left=b + t + a, color="#e9c46a", label="Parse error", edgecolor="#2d3436", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_r)
    ax.set_xlabel("Questions (of 100)")
    ax.set_title("LLM judge outcomes by judge model")
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_06_judge_win_rates_horizontal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot7_efficiency_frontier(baseline_run: dict, agentic_by_label: Dict[str, dict], out_dir: Path) -> None:
    bx = mean_per_question(baseline_run, "total_tokens")
    by = float(baseline_run["aggregate_answer_metrics"]["token_f1"])

    xs, ys, labels = [bx], [by], ["Baseline"]
    for lb, run in agentic_by_label.items():
        xs.append(mean_per_question(run, "total_tokens"))
        ys.append(float(run["aggregate_answer_metrics"]["token_f1"]))
        labels.append(f"{lb} agentic")

    fig, ax = plt.subplots(figsize=(6.6, 4.8), facecolor="white")
    ax.set_facecolor("#f6f7f9")
    colors = ["#7d8692", "#6ea8d9", "#3da89b", "#2c5364"]
    ax.scatter(xs, ys, s=[200, 240, 260, 240], c=colors[: len(xs)], edgecolors="#2d3436", linewidths=0.8, zorder=3)
    for x, y, lb in zip(xs, ys, labels):
        ax.annotate(lb, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Mean total tokens per question")
    ax.set_ylabel("Aggregate token F1")
    ax.set_title("Efficiency frontier: F1 vs tokens")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_07_f1_vs_tokens_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    report_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate seven priority story plots from report JSON.")
    parser.add_argument("--baseline-run", type=Path, default=report_dir / "run1" / "baseline_benchmark_run.json")
    parser.add_argument("--agentic-8b", type=Path, default=report_dir / "runb" / "8b" / "agentic_benchmark_run.json")
    parser.add_argument("--agentic-30b", type=Path, default=report_dir / "runb" / "30b" / "agentic_benchmark_run.json")
    parser.add_argument("--agentic-32b", type=Path, default=report_dir / "runb" / "32b" / "agentic_benchmark_run.json")
    parser.add_argument("--judge-8b", type=Path, default=report_dir / "runb" / "8b" / "llm_judge_report.json")
    parser.add_argument("--judge-30b", type=Path, default=report_dir / "runb" / "30b" / "llm_judge_report.json")
    parser.add_argument("--judge-32b", type=Path, default=report_dir / "runb" / "32b" / "llm_judge_report.json")
    parser.add_argument("--context-run-a", type=Path, default=report_dir / "runa" / "30b" / "agentic_benchmark_run.json")
    parser.add_argument("--context-run-b", type=Path, default=report_dir / "runa" / "8b" / "agentic_benchmark_run.json")
    parser.add_argument("--label-context-a", type=str, default="Run A (tight context)")
    parser.add_argument("--label-context-b", type=str, default="Run B (extended context)")
    parser.add_argument("--output-dir", type=Path, default=report_dir / "priority_report_outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_json(args.baseline_run)
    agentic = {
        "8b": load_json(args.agentic_8b),
        "30b": load_json(args.agentic_30b),
        "32b": load_json(args.agentic_32b),
    }
    judges = {
        "8b": load_json(args.judge_8b),
        "30b": load_json(args.judge_30b),
        "32b": load_json(args.judge_32b),
    }
    run_a = load_json(args.context_run_a)
    run_b = load_json(args.context_run_b)

    plot1_baseline_vs_agentic_f1(baseline, agentic, args.output_dir)
    plot2_context_ablation_f1(run_a, run_b, args.label_context_a, args.label_context_b, args.output_dir)
    plot3_f1_heatmap_baseline_and_agentic(baseline, agentic, args.output_dir)
    plot4_tool_calls_vs_f1(agentic, args.output_dir)
    plot5_stop_reason_stacked(agentic, args.output_dir)
    plot6_judge_wins_horizontal(judges, args.output_dir)
    plot7_efficiency_frontier(baseline, agentic, args.output_dir)

    print(f"Saved seven priority plots to {args.output_dir}")


if __name__ == "__main__":
    main()
