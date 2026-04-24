from __future__ import annotations

import time

from openai import OpenAI

from src.config.settings import GenerationConfig
from src.rag.generator_base import MultimodalAnswerGenerator
from src.rag.prompts import DEFAULT_GROUNDED_SYSTEM_PROMPT, build_grounded_user_prompt
from src.rag.schemas import EvidenceBundle, GeneratedAnswer


class OpenAIChatGenerator(MultimodalAnswerGenerator):
    """
    OpenAI-compatible chat generator backed by the shared remote vLLM/Modal endpoint.

    This path is intentionally text-only so both baseline and agentic benchmarks use
    the same model interface; the agentic system differs only by having tool access.
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
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        message = response.choices[0].message
        answer_text = (message.content or "").strip()
        dt = time.perf_counter() - t0

        usage = getattr(response, "usage", None)
        metadata = {
            "duration_sec": round(dt, 3),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "finish_reason": response.choices[0].finish_reason,
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
