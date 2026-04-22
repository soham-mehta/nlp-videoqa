from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrieval.schemas import RetrievedItem


class AnswerGenerator(ABC):
    """
    Placeholder interface for future answer generation.
    Kept separate from retrieval for easy swapping with agentic/tool-calling systems.
    """

    @abstractmethod
    def answer(self, query: str, evidence: list[RetrievedItem]) -> str:
        raise NotImplementedError


# TODO: implement a multimodal LLM client in a separate module.
