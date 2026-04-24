# Report figure plan

Recommended main-paper visuals:

1. `fig_overall_judge_outcomes.png`
   - Pairwise judge results: baseline wins, agentic wins, ties.

2. `fig_per_type_judge_scores.png`
   - Mean judge score by question type, using the run summary categories:
     factual, temporal, sequential, cross-modal, summary.

3. `fig_efficiency_comparison.png`
   - Baseline vs agentic on mean latency, mean total tokens, mean LLM calls, and mean tool calls.

4. `fig_tool_usage.png`
   - Agentic tool usage counts. This supports the finding that the agent mostly reduced to semantic search.

Recommended tables:

1. `table_main_results.csv`
   - Judge win rate, judge mean score, token F1, and retrieval metrics.

2. `table_efficiency.csv`
   - Latency, tokens, LLM calls, tool calls, and frames sent.

Optional qualitative support:

- `table_case_studies.csv`
  - One strong baseline-win case and one strong agentic-win case, selected automatically from the judge file.

How to run:

```bash
python generate_report_figures.py --input-dir /path/to/results --output-dir /path/to/output
```

For your current files in one directory:

```bash
python generate_report_figures.py --input-dir . --output-dir report_outputs
```

This script does not depend on seaborn and uses matplotlib defaults, which keeps the figures simple and paper-friendly.
