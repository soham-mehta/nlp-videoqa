# Non-bar report plots

This script generates four non-bar plots for the paper.

## Outputs
- `fig_nonbar_per_type_heatmap.png`
- `fig_nonbar_metric_dumbbell.png`
- `fig_nonbar_judge_score_boxplot.png`
- `fig_nonbar_tool_performance_bubble.png`

Supporting CSV files:
- `table_nonbar_per_type_heatmap.csv`
- `table_nonbar_tool_performance_bubble.csv`

## Recommended usage

```bash
python generate_nonbar_report_plots.py   --baseline-run baseline_benchmark_run.json   --judge-report-8b "llm_judge_report(2).json"   --judge-report-30b "llm_judge_report(3).json"   --judge-report-32b "llm_judge_report(1).json"   --judge-per-question-8b "llm_judge_per_question(1).jsonl"   --judge-per-question-30b "llm_judge_per_question(3).jsonl"   --judge-per-question-32b "llm_judge_per_question(2).jsonl"   --agentic-run-8b "agentic_benchmark_run(5).json"   --agentic-run-30b "agentic_benchmark_run(6).json"   --agentic-run-32b "agentic_benchmark_run(4).json"   --output-dir report_nonbar_outputs   --best-agentic-label 30b
```

## Suggested report updates
- Add the new non-bar plots to the appendix or swap them into the main paper.
- Remove the efficiency figure from the main paper and keep only the efficiency table.
- Delete the paragraph that explicitly walks through Figure 6 if that figure is removed.
- A good replacement set for the main paper is:
  1. overall judge outcomes
  2. per-type heatmap
  3. metric dumbbell plot
  4. judge-reason categories
  5. optional judge-score boxplot
