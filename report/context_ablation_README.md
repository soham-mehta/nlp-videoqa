# Context ablation appendix assets

This script generates appendix-ready assets for comparing two agentic runs that differ in context budget.

## Inputs
- Run A JSON (lower-context / failing run)
- Run B JSON (extended-context / stronger run)

## Outputs
Figures:
- `fig_context_ablation_metrics.png`
- `fig_context_ablation_stop_reasons.png`

Tables:
- `table_context_ablation_summary.csv`
- `table_context_ablation_stop_reasons.csv`
- `table_context_ablation_examples.csv`

LaTeX:
- `appendix_context_ablation_snippet.tex`

## Example usage

```bash
python generate_context_ablation_assets.py   --run-a "agentic_benchmark_run(8).json"   --run-b "agentic_benchmark_run(7).json"   --label-a "Run A (lower context)"   --label-b "Run B (extended context)"   --output-dir context_ablation_outputs
```

## Suggested file mapping from the uploaded results
- Run A (lower context / context overflow): `agentic_benchmark_run(8).json`
- Run B (extended context): `agentic_benchmark_run(7).json`

These correspond to:
- Run A token F1: ~0.124
- Run B token F1: ~0.351
