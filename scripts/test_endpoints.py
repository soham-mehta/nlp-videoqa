"""Quick health check and concurrency test for vLLM endpoints."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search indexed video chunks by semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
            },
        },
    },
]

TEST_MESSAGES = [
    {"role": "system", "content": "You answer questions about videos. Use the provided tools."},
    {"role": "user", "content": "What color is the sky? Use semantic_search to find out, then call final_answer."},
]


async def single_request(client: httpx.AsyncClient, url: str, model: str, req_id: int) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/chat/completions",
            json={
                "model": model,
                "messages": TEST_MESSAGES,
                "tools": TOOL_SCHEMAS,
                "tool_choice": "required",
                "temperature": 0.1,
                "max_tokens": 150,
            },
            timeout=120.0,
        )
        dt = time.perf_counter() - t0
        if resp.status_code == 200:
            data = resp.json()
            msg = data["choices"][0]["message"]
            tool_calls = msg.get("tool_calls", [])
            usage = data.get("usage", {})
            return {
                "req_id": req_id,
                "status": "ok",
                "latency_sec": round(dt, 2),
                "tool_calls": [tc["function"]["name"] for tc in tool_calls] if tool_calls else [],
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "content_preview": (msg.get("content") or "")[:100],
            }
        else:
            return {
                "req_id": req_id,
                "status": f"http_{resp.status_code}",
                "latency_sec": round(dt, 2),
                "error": resp.text[:200],
            }
    except Exception as e:
        dt = time.perf_counter() - t0
        return {
            "req_id": req_id,
            "status": "error",
            "latency_sec": round(dt, 2),
            "error": f"{type(e).__name__}: {e}",
        }


async def test_endpoint(url: str, model: str, concurrency_levels: list[int]) -> dict:
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{url}/models", timeout=15.0)
            if resp.status_code == 200:
                models = resp.json()
                print(f"  Health: OK — models served: {[m['id'] for m in models['data']]}")
            else:
                print(f"  Health: FAIL — status {resp.status_code}")
                return {"model": model, "status": "health_check_failed"}
        except Exception as e:
            print(f"  Health: FAIL — {e}")
            return {"model": model, "status": "unreachable", "error": str(e)}

    # Concurrency tests
    results_by_level = {}
    for n in concurrency_levels:
        print(f"\n  Concurrency={n}: sending {n} parallel requests...")
        async with httpx.AsyncClient() as client:
            t0 = time.perf_counter()
            tasks = [single_request(client, url, model, i) for i in range(n)]
            results = await asyncio.gather(*tasks)
            wall_time = time.perf_counter() - t0

        ok = [r for r in results if r["status"] == "ok"]
        failed = [r for r in results if r["status"] != "ok"]
        latencies = [r["latency_sec"] for r in ok]

        level_result = {
            "concurrency": n,
            "succeeded": len(ok),
            "failed": len(failed),
            "wall_time_sec": round(wall_time, 2),
            "mean_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "max_latency_sec": round(max(latencies), 2) if latencies else 0,
            "min_latency_sec": round(min(latencies), 2) if latencies else 0,
            "tool_call_success": sum(1 for r in ok if r.get("tool_calls")) / len(ok) if ok else 0,
            "errors": [r.get("error", "") for r in failed],
        }
        results_by_level[n] = level_result

        status = f"    {len(ok)}/{n} OK"
        if latencies:
            status += f" | wall={wall_time:.1f}s mean={level_result['mean_latency_sec']}s"
            status += f" | tool_calls={level_result['tool_call_success']*100:.0f}%"
        if failed:
            status += f" | {len(failed)} FAILED: {failed[0].get('error', '')[:80]}"
        print(status)

    return {
        "model": model,
        "url": url,
        "status": "tested",
        "concurrency_results": results_by_level,
    }


async def main_async(endpoints: list[dict], concurrency_levels: list[int]) -> list[dict]:
    all_results = []
    for ep in endpoints:
        result = await test_endpoint(ep["url"], ep["model"], concurrency_levels)
        all_results.append(result)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="dwijenchawra")
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="1,2,4,8",
        help="Comma-separated concurrency levels to test",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency_levels.split(",")]
    ws = args.workspace

    endpoints = [
        {
            "model": "Qwen/Qwen3-VL-8B-Instruct-FP8",
            "url": f"https://{ws}--nlp-videoqa-vllm-8b-serve.modal.run/v1",
        },
        {
            "model": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            "url": f"https://{ws}--nlp-videoqa-vllm-30b-serve.modal.run/v1",
        },
        {
            "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
            "url": f"https://{ws}--nlp-videoqa-vllm-32b-serve.modal.run/v1",
        },
    ]

    results = asyncio.run(main_async(endpoints, levels))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r['model']}:")
        if r["status"] != "tested":
            print(f"  Status: {r['status']}")
            continue
        for level, lr in r["concurrency_results"].items():
            print(
                f"  c={level}: {lr['succeeded']}/{level} ok, "
                f"wall={lr['wall_time_sec']}s, "
                f"mean={lr['mean_latency_sec']}s, "
                f"tools={lr['tool_call_success']*100:.0f}%"
            )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
