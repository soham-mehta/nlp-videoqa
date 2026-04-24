from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import APITimeoutError, BadRequestError, OpenAI

from src.agentic.backend import RetrievalBackend
from src.agentic.tools import TOOL_SCHEMAS, dispatch
from src.rag.image_utils import build_image_content_parts, encode_frame_base64

SYSTEM_PROMPT = """You answer questions about a multimodal video corpus.

Each video has two indexed evidence modalities with timestamps:
- text:  transcript or textual evidence associated with the video.
- frame: actual video frame images, provided inline as images.

You will be given an initial retrieval bundle. ALWAYS use at least one retrieval tool
before calling final_answer — the initial bundle is a starting point, not sufficient.

Plan retrieval based on the question:
- Factual question -> semantic_search for the specific detail asked about.
- Temporal / sequential question -> locate an anchor event with semantic_search, then
  widen with get_chunks_by_timestamp or get_nearby_chunks.
- Cross-modal question -> search one modality to find a timestamp, then search
  the other modality at that same timestamp.

IMPORTANT: You MUST call tools using the provided function calling interface.
Do NOT write tool calls as text or code. Use the function calling mechanism.

When you have enough evidence, call final_answer with the answer text and
the evidence_chunk_ids you relied on."""


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
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    num_llm_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_chunk_ids": self.evidence_chunk_ids,
            "prompt_text": self.prompt_text,
            "total_tool_calls": self.total_tool_calls,
            "stopped_reason": self.stopped_reason,
            "trajectory": [asdict(s) for s in self.trajectory],
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "num_llm_calls": self.num_llm_calls,
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
            "First, use semantic_search to find evidence directly relevant to the question.",
            "Then call final_answer with the answer and the evidence chunk ids you used.",
        ]
    )
    return "\n".join(lines)


def _extract_frame_paths_from_tool_result(result: str) -> list[str]:
    try:
        chunks = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(chunks, list):
        return []
    paths: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, dict) and chunk.get("frame_path"):
            paths.append(chunk["frame_path"])
    return paths


def run_agent(
    question: str,
    backend: RetrievalBackend,
    client: OpenAI,
    model: str,
    video_id: str | None = None,
    initial_context: str | None = None,
    initial_frame_paths: list[str] | None = None,
    max_tool_calls: int = 8,
    temperature: float = 0.2,
) -> AgentResult:
    user_prompt = _build_user_prompt(
        question=question,
        video_id=video_id,
        initial_context=initial_context,
    )

    max_images = 8
    user_content: Any
    image_parts = build_image_content_parts(initial_frame_paths or [])[:max_images]
    total_images = len(image_parts)
    if image_parts:
        user_content = [{"type": "text", "text": user_prompt}, *image_parts]
    else:
        user_content = user_prompt

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    trajectory: list[TrajectoryStep] = []
    accum_prompt_tokens = 0
    accum_completion_tokens = 0
    accum_llm_calls = 0

    while True:
        force_final = len(trajectory) >= max_tool_calls
        if force_final:
            tool_choice: Any = {"type": "function", "function": {"name": "final_answer"}}
        else:
            tool_choice = "required"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice=tool_choice,
                temperature=temperature,
            )
        except (BadRequestError, APITimeoutError) as e:
            last_tool_text = ""
            for step in reversed(trajectory):
                if step.result:
                    last_tool_text = step.result[:500]
                    break
            reason = "api_timeout" if isinstance(e, APITimeoutError) else "context_overflow"
            return AgentResult(
                answer=last_tool_text or f"({reason} — no answer produced)",
                evidence_chunk_ids=[s.args.get("evidence_chunk_ids", []) for s in trajectory if s.tool == "final_answer"][:1] or [],
                prompt_text=user_prompt,
                trajectory=trajectory,
                total_tool_calls=len(trajectory),
                stopped_reason=reason,
                total_prompt_tokens=accum_prompt_tokens,
                total_completion_tokens=accum_completion_tokens,
                num_llm_calls=accum_llm_calls,
            )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        usage = getattr(resp, "usage", None)
        accum_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        accum_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        accum_llm_calls += 1

        if not msg.tool_calls:
            return AgentResult(
                answer=msg.content or "",
                evidence_chunk_ids=[],
                prompt_text=user_prompt,
                trajectory=trajectory,
                total_tool_calls=len(trajectory),
                stopped_reason="no_tool_call",
                total_prompt_tokens=accum_prompt_tokens,
                total_completion_tokens=accum_completion_tokens,
                num_llm_calls=accum_llm_calls,
            )

        tool_frame_paths: list[str] = []
        # If final_answer is the only tool call, honor it; otherwise skip it
        # so the model can't short-circuit by bundling final_answer with retrieval tools.
        has_non_final = any(tc.function.name != "final_answer" for tc in msg.tool_calls)
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "final_answer":
                if has_non_final:
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": '{"status": "deferred — finish retrieval first"}'}
                    )
                    continue
                return AgentResult(
                    answer=str(args.get("answer", "")),
                    evidence_chunk_ids=list(args.get("evidence_chunk_ids", [])),
                    prompt_text=user_prompt,
                    trajectory=trajectory,
                    total_tool_calls=len(trajectory),
                    stopped_reason="max_tool_calls" if force_final else "final_answer",
                    total_prompt_tokens=accum_prompt_tokens,
                    total_completion_tokens=accum_completion_tokens,
                    num_llm_calls=accum_llm_calls,
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
            tool_frame_paths.extend(_extract_frame_paths_from_tool_result(result))

        remaining_slots = max_images - total_images
        if remaining_slots > 0 and tool_frame_paths:
            tool_images = build_image_content_parts(tool_frame_paths)[:remaining_slots]
            if tool_images:
                total_images += len(tool_images)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Frame images from the retrieval results above:"},
                        *tool_images,
                    ],
                })
