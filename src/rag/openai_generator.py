from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from src.config.settings import GenerationConfig
from src.rag.generator_base import MultimodalAnswerGenerator
from src.rag.image_utils import build_image_content_parts
from src.rag.prompts import DEFAULT_GROUNDED_SYSTEM_PROMPT, build_grounded_user_prompt
from src.rag.schemas import EvidenceBundle, GeneratedAnswer


def _build_user_content(text_prompt: str, evidence: EvidenceBundle) -> list[dict] | str:
    frame_paths = [f.frame_path for f in evidence.frames if f.frame_path]
    image_parts = build_image_content_parts(frame_paths)
    if not image_parts:
        return text_prompt
    return [{"type": "text", "text": text_prompt}, *image_parts]


class OpenAIChatGenerator(MultimodalAnswerGenerator):
    """
    OpenAI-compatible chat generator. Sends frame images as base64 when available,
    falling back to text-only when no frames exist or files are missing.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_sec,
        )

    def generate_answer(
        self,
        question: str,
        retrieved_evidence: EvidenceBundle,
        system_prompt: str | None = None,
    ) -> GeneratedAnswer:
        t0 = time.perf_counter()
        effective_system_prompt = system_prompt or DEFAULT_GROUNDED_SYSTEM_PROMPT
        user_prompt = build_grounded_user_prompt(question=question, evidence=retrieved_evidence)
        user_content: Any = _build_user_content(user_prompt, retrieved_evidence)
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        message = response.choices[0].message
        answer_text = (message.content or "").strip()
        dt = time.perf_counter() - t0

        usage = getattr(response, "usage", None)
        num_frames_sent = (
            sum(1 for p in user_content if isinstance(p, dict) and p.get("type") == "image_url")
            if isinstance(user_content, list)
            else 0
        )
        metadata = {
            "duration_sec": round(dt, 3),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "finish_reason": response.choices[0].finish_reason,
            "num_frames_sent": num_frames_sent,
        }

        return GeneratedAnswer(
            answer_text=answer_text,
            model_name=self.config.model_name,
            prompt_text=user_prompt,
            used_evidence_ids=[x.evidence_id for x in retrieved_evidence.transcripts]
            + [x.evidence_id for x in retrieved_evidence.frames],
            raw_response_text=answer_text,
            metadata=metadata,
        )
