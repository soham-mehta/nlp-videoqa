from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from src.agentic.backend import RetrievalBackend
from src.agentic.tools import TOOL_SCHEMAS, dispatch

SYSTEM_PROMPT = """You answer questions about a multimodal video corpus.

Each video has two indexed evidence modalities with timestamps:
- text:  transcript or textual evidence associated with the video.
- frame: frame evidence represented by frame ids and paths.

You will be given an initial retrieval bundle from the same baseline system used in the
non-agentic benchmark. Start from that evidence, and only use tools when you need more.

You have retrieval tools over the same shared index. Plan retrieval based on the question:
- Factual question -> one semantic_search is often enough.
- Temporal / sequential question -> locate an anchor event with a search, then
  widen with get_chunks_by_timestamp or get_nearby_chunks.
- Cross-modal question -> search one modality to find a timestamp, then search
  the other modality at that same timestamp.

Keep tool calls efficient. When you have enough evidence, call final_answer and
list the chunk ids you actually relied on."""


@dataclass
class TrajectoryStep:
    tool: str
    args: dict
    result: str
    latency_sec: float


@dataclass
class AgentResult:
    answer: str
    evidence_chunk_ids: list[str]
    prompt_text: str
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    total_tool_calls: int = 0
    stopped_reason: str = "final_answer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_chunk_ids": self.evidence_chunk_ids,
            "prompt_text": self.prompt_text,
            "total_tool_calls": self.total_tool_calls,
            "stopped_reason": self.stopped_reason,
            "trajectory": [asdict(s) for s in self.trajectory],
        }


def _build_user_prompt(question: str, video_id: str | None, initial_context: str | None) -> str:
    lines = ["Question:", question.strip()]
    if video_id:
        lines.extend(["", f"Target video_id: {video_id}"])
    if initial_context:
        lines.extend(["", "Initial retrieved evidence:", initial_context.strip()])
    lines.extend(
        [
            "",
            "Use the tools only if the initial evidence is insufficient.",
            "When you are done, call final_answer with the answer and the evidence chunk ids you used.",
        ]
    )
    return "\n".join(lines)


def run_agent(
    question: str,
    backend: RetrievalBackend,
    client: OpenAI,
    model: str,
    video_id: str | None = None,
    initial_context: str | None = None,
    max_tool_calls: int = 8,
    temperature: float = 0.2,
) -> AgentResult:
    user_prompt = _build_user_prompt(
        question=question,
        video_id=video_id,
        initial_context=initial_context,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    trajectory: list[TrajectoryStep] = []

    while True:
        force_final = len(trajectory) >= max_tool_calls
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice=(
                {"type": "function", "function": {"name": "final_answer"}}
                if force_final
                else "auto"
            ),
            temperature=temperature,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return AgentResult(
                answer=msg.content or "",
                evidence_chunk_ids=[],
                prompt_text=user_prompt,
                trajectory=trajectory,
                total_tool_calls=len(trajectory),
                stopped_reason="no_tool_call",
            )

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "final_answer":
                return AgentResult(
                    answer=str(args.get("answer", "")),
                    evidence_chunk_ids=list(args.get("evidence_chunk_ids", [])),
                    prompt_text=user_prompt,
                    trajectory=trajectory,
                    total_tool_calls=len(trajectory),
                    stopped_reason="max_tool_calls" if force_final else "final_answer",
                )

            t0 = time.perf_counter()
            try:
                result = dispatch(backend, name, args)
            except Exception as e:
                result = json.dumps({"error": f"{type(e).__name__}: {e}"})
            latency = time.perf_counter() - t0

            trajectory.append(
                TrajectoryStep(tool=name, args=args, result=result, latency_sec=latency)
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
