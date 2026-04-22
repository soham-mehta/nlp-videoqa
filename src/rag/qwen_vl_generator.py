from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.config.settings import GenerationConfig
from src.rag.generator_base import MultimodalAnswerGenerator
from src.rag.prompts import DEFAULT_GROUNDED_SYSTEM_PROMPT, build_grounded_user_prompt
from src.rag.schemas import EvidenceBundle, GeneratedAnswer


class QwenVLGenerator(MultimodalAnswerGenerator):
    """
    Baseline multimodal generator for Qwen2.5-VL models.
    Interface remains backend-agnostic via MultimodalAnswerGenerator.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name, torch_dtype="auto"
        )
        self.model.to(torch.device(config.device))
        self.model.eval()

    def _build_messages(
        self,
        question: str,
        evidence: EvidenceBundle,
        system_prompt: str | None,
    ) -> tuple[list[dict[str, object]], list[Image.Image], str]:
        effective_system_prompt = system_prompt or DEFAULT_GROUNDED_SYSTEM_PROMPT
        user_prompt = build_grounded_user_prompt(question=question, evidence=evidence)

        images: list[Image.Image] = []
        user_content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]
        for frame in evidence.frames:
            frame_path = Path(frame.frame_path)
            if not frame_path.exists():
                continue
            try:
                images.append(Image.open(frame_path).convert("RGB"))
                user_content.append({"type": "image"})
            except Exception:
                continue

        messages: list[dict[str, object]] = [
            {"role": "system", "content": [{"type": "text", "text": effective_system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        return messages, images, user_prompt

    def generate_answer(
        self,
        question: str,
        retrieved_evidence: EvidenceBundle,
        system_prompt: str | None = None,
    ) -> GeneratedAnswer:
        t0 = time.perf_counter()
        messages, images, user_prompt = self._build_messages(
            question=question,
            evidence=retrieved_evidence,
            system_prompt=system_prompt,
        )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
            )

        input_token_count = int(model_inputs["input_ids"].shape[1])
        completion_ids = generated[:, input_token_count:]
        decoded = self.processor.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
        raw_text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        dt = time.perf_counter() - t0

        return GeneratedAnswer(
            answer_text=decoded,
            model_name=self.config.model_name,
            prompt_text=user_prompt,
            used_evidence_ids=[x.evidence_id for x in retrieved_evidence.transcripts]
            + [x.evidence_id for x in retrieved_evidence.frames],
            raw_response_text=raw_text,
            metadata={"duration_sec": round(dt, 3), "num_images": len(images)},
        )
