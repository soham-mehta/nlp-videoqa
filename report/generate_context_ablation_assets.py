
from __future__ import annotations

import argparse
import filecmp
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

_REPORT_DIR = Path(__file__).resolve().parent
_README_DEFAULT_INPUT_DIR = _REPORT_DIR / "context_ablation_inputs"
# README: Run A = (8) lower-context; Run B = (7) extended-context.
_README_DEFAULT_RUN_A = _README_DEFAULT_INPUT_DIR / "agentic_benchmark_run(8).json"
_README_DEFAULT_RUN_B = _README_DEFAULT_INPUT_DIR / "agentic_benchmark_run(7).json"
# Checked-in ablation pair (same benchmark; different agent configs): ~0.124 vs ~0.351 token F1.
_RUNA_FALLBACK_RUN_A = _REPORT_DIR / "runa" / "30b" / "agentic_benchmark_run.json"
_RUNA_FALLBACK_RUN_B = _REPORT_DIR / "runa" / "8b" / "agentic_benchmark_run.json"


def resolve_readme_default_paths() -> tuple[Path, Path, str | None]:
    """Prefer distinct (8)/(7) under context_ablation_inputs; else use report/runa/*."""
    p8, p7 = _README_DEFAULT_RUN_A, _README_DEFAULT_RUN_B
    if p8.is_file() and p7.is_file() and not filecmp.cmp(p8, p7, shallow=False):
        return p8, p7, None
    if _RUNA_FALLBACK_RUN_A.is_file() and _RUNA_FALLBACK_RUN_B.is_file():
        return _RUNA_FALLBACK_RUN_A, _RUNA_FALLBACK_RUN_B, (
            "Using report/runa/30b and report/runa/8b agentic_benchmark_run.json "
            f"(distinct uploads). Place real {p8.name} / {p7.name} under {_README_DEFAULT_INPUT_DIR} to override."
        )
    return p8, p7, None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_efficiency(run: dict) -> dict:
    rows = []
    stop_reasons = Counter()
    for item in run.get("per_question", []):
        pq = item.get("per_query_metrics", {})
        gm = item.get("debug_info", {}).get("generator_metadata", {})
        rd = item.get("debug_info", {}).get("run_debug", {})
        stop_reason = gm.get("agent_stopped_reason") or rd.get("agent_stopped_reason") or "unknown"
        stop_reasons[stop_reason] += 1
        rows.append(
            {
                "latency_sec": pq.get("latency_sec", 0.0),
                "prompt_tokens": pq.get("prompt_tokens", 0.0),
                "completion_tokens": pq.get("completion_tokens", 0.0),
                "total_tokens": pq.get("total_tokens", 0.0),
                "num_llm_calls": pq.get("num_llm_calls", 0.0),
                "num_tool_calls": pq.get("num_tool_calls", 0.0),
                "num_frames_sent": pq.get("num_frames_sent", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"stop_reasons": dict(stop_reasons)}
    return {
        "mean_latency_sec": df["latency_sec"].mean(),
        "mean_prompt_tokens": df["prompt_tokens"].mean(),
        "mean_completion_tokens": df["completion_tokens"].mean(),
        "mean_total_tokens": df["total_tokens"].mean(),
        "total_tokens": df["total_tokens"].sum(),
        "mean_llm_calls": df["num_llm_calls"].mean(),
        "mean_tool_calls": df["num_tool_calls"].mean(),
        "mean_frames_sent": df["num_frames_sent"].mean(),
        "stop_reasons": dict(stop_reasons),
    }


def find_question(run: dict, question_id: str) -> dict | None:
    for item in run.get("per_question", []):
        if item.get("question_id") == question_id:
            return item
    return None


def build_summary_table(run_a: dict, run_b: dict, label_a: str, label_b: str, output_dir: Path) -> pd.DataFrame:
    eff_a = aggregate_efficiency(run_a)
    eff_b = aggregate_efficiency(run_b)

    rows = [
        {
            "run": label_a,
            "token_f1": run_a["aggregate_answer_metrics"]["token_f1"],
            "top_k_hit": run_a["aggregate_retrieval_metrics"]["top_k_hit"],
            "evidence_overlap_hit": run_a["aggregate_retrieval_metrics"]["evidence_overlap_hit"],
            "evidence_recall_proxy": run_a["aggregate_retrieval_metrics"]["evidence_recall_proxy"],
            "mean_total_tokens": eff_a.get("mean_total_tokens", 0.0),
            "mean_llm_calls": eff_a.get("mean_llm_calls", 0.0),
            "mean_tool_calls": eff_a.get("mean_tool_calls", 0.0),
            "dominant_stop_reason": max(eff_a["stop_reasons"], key=eff_a["stop_reasons"].get) if eff_a["stop_reasons"] else "unknown",
        },
        {
            "run": label_b,
            "token_f1": run_b["aggregate_answer_metrics"]["token_f1"],
            "top_k_hit": run_b["aggregate_retrieval_metrics"]["top_k_hit"],
            "evidence_overlap_hit": run_b["aggregate_retrieval_metrics"]["evidence_overlap_hit"],
            "evidence_recall_proxy": run_b["aggregate_retrieval_metrics"]["evidence_recall_proxy"],
            "mean_total_tokens": eff_b.get("mean_total_tokens", 0.0),
            "mean_llm_calls": eff_b.get("mean_llm_calls", 0.0),
            "mean_tool_calls": eff_b.get("mean_tool_calls", 0.0),
            "dominant_stop_reason": max(eff_b["stop_reasons"], key=eff_b["stop_reasons"].get) if eff_b["stop_reasons"] else "unknown",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_context_ablation_summary.csv", index=False)
    return df


def plot_metric_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    metric_order = ["token_f1", "top_k_hit", "evidence_overlap_hit", "evidence_recall_proxy"]
    pretty = {
        "token_f1": "Token F1",
        "top_k_hit": "Top-k hit",
        "evidence_overlap_hit": "Overlap hit",
        "evidence_recall_proxy": "Recall proxy",
    }

    x = list(range(len(metric_order)))
    width = 0.38

    plt.figure(figsize=(8.2, 4.8))
    plt.bar([i - width / 2 for i in x], [summary_df.iloc[0][m] for m in metric_order], width=width, label=summary_df.iloc[0]["run"])
    plt.bar([i + width / 2 for i in x], [summary_df.iloc[1][m] for m in metric_order], width=width, label=summary_df.iloc[1]["run"])
    plt.xticks(x, [pretty[m] for m in metric_order], rotation=20, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Context-budget ablation: retrieval and answer metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_context_ablation_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_stop_reasons(run_a: dict, run_b: dict, label_a: str, label_b: str, output_dir: Path) -> pd.DataFrame:
    eff_a = aggregate_efficiency(run_a)
    eff_b = aggregate_efficiency(run_b)
    all_reasons = sorted(set(eff_a["stop_reasons"].keys()) | set(eff_b["stop_reasons"].keys()))

    vals_a = [eff_a["stop_reasons"].get(r, 0) for r in all_reasons]
    vals_b = [eff_b["stop_reasons"].get(r, 0) for r in all_reasons]
    x = list(range(len(all_reasons)))
    width = 0.38

    plt.figure(figsize=(8.4, 4.8))
    plt.bar([i - width / 2 for i in x], vals_a, width=width, label=label_a)
    plt.bar([i + width / 2 for i in x], vals_b, width=width, label=label_b)
    plt.xticks(x, all_reasons, rotation=20, ha="right")
    plt.ylabel("Number of questions")
    plt.title("Context-budget ablation: stop reasons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_context_ablation_stop_reasons.png", dpi=300, bbox_inches="tight")
    plt.close()

    stop_df = pd.DataFrame({
        "stop_reason": all_reasons,
        label_a: vals_a,
        label_b: vals_b,
    })
    stop_df.to_csv(output_dir / "table_context_ablation_stop_reasons.csv", index=False)
    return stop_df


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def write_appendix_snippet(
    summary_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    """Minimal LaTeX snippet matching report/context_ablation_README.md."""
    lines: list[str] = [
        "% Auto-generated by generate_context_ablation_assets.py",
        "% Context-window ablation summary",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{Context ablation: {_latex_escape(label_a)} vs {_latex_escape(label_b)}.}}",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Run & Token F1 & Top-$k$ & Overlap & Recall proxy & Mean tokens & Dominant stop \\\\",
        "\\hline",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{_latex_escape(str(row['run']))} & "
            f"{row['token_f1']:.4f} & {row['top_k_hit']:.4f} & {row['evidence_overlap_hit']:.4f} & "
            f"{row['evidence_recall_proxy']:.4f} & {row['mean_total_tokens']:.1f} & "
            f"{_latex_escape(str(row['dominant_stop_reason']))} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])

    col_a, col_b = stop_df.columns[1], stop_df.columns[2]
    lines.extend(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\caption{Stop reasons (question counts).}",
            "\\begin{tabular}{lrr}",
            "\\hline",
            f"Reason & {_latex_escape(col_a)} & {_latex_escape(col_b)} \\\\",
            "\\hline",
        ]
    )
    for _, row in stop_df.iterrows():
        lines.append(
            f"{_latex_escape(str(row['stop_reason']))} & {int(row[col_a])} & {int(row[col_b])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])

    out_path = output_dir / "appendix_context_ablation_snippet.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_case_table(run_a: dict, run_b: dict, label_a: str, label_b: str, question_ids: List[str], output_dir: Path) -> pd.DataFrame:
    rows = []
    for qid in question_ids:
        a = find_question(run_a, qid)
        b = find_question(run_b, qid)
        if not a or not b:
            continue

        stop_a = a.get("debug_info", {}).get("generator_metadata", {}).get("agent_stopped_reason") or a.get("debug_info", {}).get("run_debug", {}).get("agent_stopped_reason") or "unknown"
        stop_b = b.get("debug_info", {}).get("generator_metadata", {}).get("agent_stopped_reason") or b.get("debug_info", {}).get("run_debug", {}).get("agent_stopped_reason") or "unknown"

        rows.append(
            {
                "question_id": qid,
                "question": a.get("debug_info", {}).get("question", ""),
                "gold_answer": a.get("debug_info", {}).get("gold_answer", ""),
                f"{label_a}_answer": a.get("final_answer", ""),
                f"{label_a}_token_f1": a.get("answer_metrics", {}).get("token_f1", 0.0),
                f"{label_a}_stop_reason": stop_a,
                f"{label_b}_answer": b.get("final_answer", ""),
                f"{label_b}_token_f1": b.get("answer_metrics", {}).get("token_f1", 0.0),
                f"{label_b}_stop_reason": stop_b,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_context_ablation_examples.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate appendix assets for the context-window ablation.")
    parser.add_argument("--run-a", type=Path, default=None, help="Lower-context / failing run JSON (see README).")
    parser.add_argument("--run-b", type=Path, default=None, help="Extended-context / stronger run JSON (see README).")
    parser.add_argument(
        "--use-readme-default-inputs",
        action="store_true",
        help=(
            f"Prefer {_README_DEFAULT_RUN_A.name} and {_README_DEFAULT_RUN_B.name} under {_README_DEFAULT_INPUT_DIR} "
            "when both exist and differ; otherwise use report/runa/30b vs report/runa/8b agentic_benchmark_run.json."
        ),
    )
    parser.add_argument("--label-a", type=str, default="Run A (lower context)")
    parser.add_argument("--label-b", type=str, default="Run B (extended context)")
    parser.add_argument("--output-dir", type=Path, default=Path("context_ablation_outputs"))
    parser.add_argument("--question-ids", nargs="*", default=["cx_f_001", "cx_f_002"])
    args = parser.parse_args()

    run_a_path = args.run_a
    run_b_path = args.run_b
    default_note: str | None = None
    if args.use_readme_default_inputs:
        run_a_path, run_b_path, default_note = resolve_readme_default_paths()
    if run_a_path is None or run_b_path is None:
        print(
            "Provide --run-a and --run-b, or use --use-readme-default-inputs "
            f"(distinct files under {_README_DEFAULT_INPUT_DIR}, or report/runa/30b and runa/8b).",
            file=sys.stderr,
        )
        sys.exit(2)

    if not run_a_path.is_file() or not run_b_path.is_file():
        print(
            "Missing input JSON. Tried:\n"
            f"  {_README_DEFAULT_RUN_A}\n  {_README_DEFAULT_RUN_B}\n"
            f"  {_RUNA_FALLBACK_RUN_A}\n  {_RUNA_FALLBACK_RUN_B}\n"
            "Pass explicit --run-a/--run-b, or add the files above (see report/context_ablation_README.md).",
            file=sys.stderr,
        )
        sys.exit(1)
    if default_note:
        print(default_note, file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_a = load_json(run_a_path)
    run_b = load_json(run_b_path)

    summary_df = build_summary_table(run_a, run_b, args.label_a, args.label_b, args.output_dir)
    plot_metric_comparison(summary_df, args.output_dir)
    stop_df = plot_stop_reasons(run_a, run_b, args.label_a, args.label_b, args.output_dir)
    build_case_table(run_a, run_b, args.label_a, args.label_b, args.question_ids, args.output_dir)
    write_appendix_snippet(summary_df, stop_df, args.label_a, args.label_b, args.output_dir)

    if filecmp.cmp(run_a_path, run_b_path, shallow=False):
        print(
            "WARNING: --run-a and --run-b are byte-identical. Replace with distinct ablation JSONs "
            "from context_ablation_README.md for meaningful figures.",
            file=sys.stderr,
        )

    print(f"Saved context ablation assets to {args.output_dir}")


if __name__ == "__main__":
    main()
