from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI
from src.utils.io import write_json, write_jsonl

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating answers to questions about video content.

You will be given:
- A question about a video
- The correct reference answer
- Two candidate answers (Answer A and Answer B) from different systems

Your task: decide which answer is better based on correctness, completeness, and relevance compared to the reference answer.

Respond with EXACTLY this JSON format (no other text):
{
  "reasoning": "<2-3 sentences explaining your judgment>",
  "winner": "<A, B, or tie>",
  "score_a": <1-5>,
  "score_b": <1-5>
}

Scoring rubric:
1 = Completely wrong or irrelevant
2 = Mentions the right topic but key facts are wrong
3 = Partially correct, missing important details
4 = Mostly correct with minor omissions or inaccuracies
5 = Fully correct and complete"""


def build_judge_prompt(question: str, gold: str, answer_a: str, answer_b: str) -> str:
    return f"""Question: {question}

Reference answer: {gold}

Answer A: {answer_a}

Answer B: {answer_b}

Judge which answer is better. Respond with JSON only."""


def parse_judge_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"reasoning": text, "winner": "parse_error", "score_a": 0, "score_b": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise LLM-as-judge comparison of two systems.")
    parser.add_argument("--benchmark-path", type=Path, default=Path("data/benchmark/multimodal_benchmark_v2.json"))
    parser.add_argument("--baseline-predictions", type=Path, default=Path("data/eval/baseline_predictions_v1.jsonl"))
    parser.add_argument("--agentic-predictions", type=Path, default=Path("data/eval/agentic_predictions_v1.jsonl"))
    parser.add_argument("--generation-base-url", type=str, default=None)
    parser.add_argument("--generation-api-key", type=str, default="not-needed")
    parser.add_argument("--generation-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-json", type=Path, default=Path("data/eval/llm_judge_report.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/eval/llm_judge_per_question.jsonl"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for position randomization.")
    args = parser.parse_args()

    with open(args.benchmark_path) as f:
        benchmark = json.load(f)
    gold_by_id = {}
    for video in benchmark:
        vid = video["video_id"]
        for q in video["items"]:
            q["video_id"] = vid
            gold_by_id[q["question_id"]] = q

    with open(args.baseline_predictions) as f:
        baseline_rows = {json.loads(l)["question_id"]: json.loads(l) for l in f}
    with open(args.agentic_predictions) as f:
        agentic_rows = {json.loads(l)["question_id"]: json.loads(l) for l in f}

    client = OpenAI(
        base_url=args.generation_base_url,
        api_key=args.generation_api_key,
        timeout=120.0,
    )

    rng = random.Random(args.seed)
    question_ids = sorted(gold_by_id.keys())
    results = []
    wins = {"baseline": 0, "agentic": 0, "tie": 0, "parse_error": 0}
    total_score_baseline = 0.0
    total_score_agentic = 0.0

    for i, qid in enumerate(question_ids):
        gold_item = gold_by_id[qid]
        question = gold_item["question"]
        gold_answer = gold_item["ideal_answer"]
        baseline_answer = baseline_rows[qid]["final_answer"]
        agentic_answer = agentic_rows[qid]["final_answer"]

        # Randomize position to avoid position bias
        baseline_is_a = rng.random() < 0.5
        if baseline_is_a:
            answer_a, answer_b = baseline_answer, agentic_answer
            label_a, label_b = "baseline", "agentic"
        else:
            answer_a, answer_b = agentic_answer, baseline_answer
            label_a, label_b = "agentic", "baseline"

        user_prompt = build_judge_prompt(question, gold_answer, answer_a, answer_b)

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=args.generation_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                max_tokens=300,
            )
            raw_text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        except Exception as e:
            raw_text = f"ERROR: {e}"
            prompt_tokens = 0
            completion_tokens = 0
        latency = time.perf_counter() - t0

        parsed = parse_judge_response(raw_text)

        # Map back from A/B to baseline/agentic
        raw_winner = parsed.get("winner", "parse_error").strip().upper()
        score_a = parsed.get("score_a", 0)
        score_b = parsed.get("score_b", 0)

        if raw_winner == "A":
            winner = label_a
            score_baseline = score_a if baseline_is_a else score_b
            score_agentic = score_b if baseline_is_a else score_a
        elif raw_winner == "B":
            winner = label_b
            score_baseline = score_a if baseline_is_a else score_b
            score_agentic = score_b if baseline_is_a else score_a
        elif raw_winner == "TIE":
            winner = "tie"
            score_baseline = score_a if baseline_is_a else score_b
            score_agentic = score_b if baseline_is_a else score_a
        else:
            winner = "parse_error"
            score_baseline = score_a if baseline_is_a else score_b
            score_agentic = score_b if baseline_is_a else score_a

        wins[winner] = wins.get(winner, 0) + 1
        total_score_baseline += score_baseline
        total_score_agentic += score_agentic

        row = {
            "question_id": qid,
            "video_id": gold_item.get("video_id", ""),
            "question_type": gold_item.get("question_type", ""),
            "winner": winner,
            "score_baseline": score_baseline,
            "score_agentic": score_agentic,
            "reasoning": parsed.get("reasoning", ""),
            "baseline_is_position_a": baseline_is_a,
            "latency_sec": round(latency, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        results.append(row)
        status = f"[{i+1}/{len(question_ids)}] {qid}: winner={winner} baseline={score_baseline} agentic={score_agentic}"
        print(status)

    n = len(results)
    # Per question type breakdown
    type_stats: dict[str, dict] = {}
    for r in results:
        qt = r["question_type"]
        if qt not in type_stats:
            type_stats[qt] = {"count": 0, "baseline_wins": 0, "agentic_wins": 0, "ties": 0,
                              "baseline_score_sum": 0.0, "agentic_score_sum": 0.0}
        ts = type_stats[qt]
        ts["count"] += 1
        ts["baseline_score_sum"] += r["score_baseline"]
        ts["agentic_score_sum"] += r["score_agentic"]
        if r["winner"] == "baseline":
            ts["baseline_wins"] += 1
        elif r["winner"] == "agentic":
            ts["agentic_wins"] += 1
        elif r["winner"] == "tie":
            ts["ties"] += 1

    per_type_summary = {}
    for qt, ts in sorted(type_stats.items()):
        per_type_summary[qt] = {
            "count": ts["count"],
            "baseline_wins": ts["baseline_wins"],
            "agentic_wins": ts["agentic_wins"],
            "ties": ts["ties"],
            "mean_score_baseline": round(ts["baseline_score_sum"] / ts["count"], 3),
            "mean_score_agentic": round(ts["agentic_score_sum"] / ts["count"], 3),
        }

    report = {
        "num_questions": n,
        "wins": wins,
        "win_rate_baseline": round(wins["baseline"] / n, 3),
        "win_rate_agentic": round(wins["agentic"] / n, 3),
        "tie_rate": round(wins["tie"] / n, 3),
        "mean_score_baseline": round(total_score_baseline / n, 3),
        "mean_score_agentic": round(total_score_agentic / n, 3),
        "per_question_type": per_type_summary,
        "judge_model": args.generation_model,
        "seed": args.seed,
    }

    write_json(args.output_json, report)
    write_jsonl(args.output_jsonl, results)
    print("\n" + json.dumps(report, indent=2))
    print(f"\nSaved report to: {args.output_json}")
    print(f"Saved per-question results to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
