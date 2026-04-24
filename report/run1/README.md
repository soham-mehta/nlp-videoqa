# Run 1: Baseline RAG vs Agentic RAG (tool_choice="auto" after first call)

**Date:** 2026-04-24
**Model:** Qwen/Qwen2.5-VL-7B-Instruct (vLLM on Modal, L4 GPU)
**Benchmark:** multimodal_benchmark_v2.json (100 questions, 10 videos, 5 question types)
**Retrieval:** Modal FAISS index (SigLIP2, 27,126 items), top_k=8, max_text=4, max_frame=4

## Agent Configuration

- `tool_choice="required"` on first LLM call only, `"auto"` thereafter
- `max_tool_calls=8`, `temperature=0.2`
- vLLM `max_model_len=8192`, tool-call-parser: `hermes` (changed from `qwen25` mid-run)
- Context overflow handled gracefully (1 empty answer: jg_sum_001)

## Key Results

### Answer Quality (Token F1)
| Metric        | Baseline | Agentic | Delta  |
|---------------|----------|---------|--------|
| Overall F1    | 0.296    | 0.293   | -0.003 |
| Factoid       | 0.344    | 0.350   | +0.006 |
| Multimodal    | 0.292    | 0.303   | +0.011 |
| Procedure     | 0.241    | 0.255   | +0.014 |
| Temporal      | 0.310    | 0.262   | -0.048 |
| Other         | 0.294    | 0.294   | +0.000 |

### LLM-as-Judge (Qwen2.5-VL-7B, pairwise comparison)
| Outcome        | Count | Rate |
|----------------|-------|------|
| Baseline wins  | 64    | 64%  |
| Agentic wins   | 30    | 30%  |
| Ties           | 6     | 6%   |

Mean judge score: baseline=3.91/5, agentic=2.98/5

**Per question type (judge):**
- Agentic strongest on **summary** (9 wins, mean 3.85 vs 3.65)
- Agentic weakest on **cross_modal** (5 wins) and **sequential** (4 wins)

### Efficiency
| Metric         | Baseline | Agentic |
|----------------|----------|---------|
| Mean latency   | 6.6s     | 17.6s   |
| Mean tokens    | 1,591    | 5,887   |
| Total tokens   | 159K     | 589K    |
| Mean LLM calls | 1.0      | 1.93    |
| Mean tool calls| 0        | 1.17    |

### Tool Usage Analysis
The agent almost exclusively used `semantic_search` (111/117 calls). Only 6 calls to `get_chunks_by_timestamp`. Zero calls to `get_nearby_chunks` or `get_video_metadata`.

**Root cause:** `tool_choice="auto"` after the first forced call caused the 7B model to respond with text instead of making additional tool calls. 93/100 questions stopped with `no_tool_call` after a single semantic_search.

## Files
| File | Description |
|------|-------------|
| `baseline_benchmark_run.json` | Full baseline results with per-question metrics |
| `baseline_predictions_v1.jsonl` | 100 baseline prediction rows |
| `baseline_detail.jsonl` | Per-question efficiency metrics (baseline) |
| `baseline_progress.txt` | Real-time progress log from baseline run |
| `agentic_benchmark_run.json` | Full agentic results with trajectories |
| `agentic_predictions_v1.jsonl` | 100 agentic prediction rows |
| `agentic_detail.jsonl` | Per-question efficiency metrics (agentic) |
| `llm_judge_report.json` | Aggregate judge metrics and per-type breakdown |
| `llm_judge_per_question.jsonl` | Per-question judge scores, winners, and reasoning |

## Caveat
Same model (Qwen2.5-VL-7B) generated answers and served as judge. Possible self-preference bias. Position randomization (seed=42) showed mild position-A bias (72% vs 56% baseline win rate) but baseline won in both positions. Token F1 vs judge correlation was weak (r=0.215).
