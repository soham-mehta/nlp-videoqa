from __future__ import annotations

from abc import ABC, abstractmethod

from src.rag.schemas import EvidenceBundle, GeneratedAnswer


class MultimodalAnswerGenerator(ABC):
    @abstractmethod
    def generate_answer(
        self,
        question: str,
        retrieved_evidence: EvidenceBundle,
        system_prompt: str | None = None,
    ) -> GeneratedAnswer:
        raise NotImplementedError
