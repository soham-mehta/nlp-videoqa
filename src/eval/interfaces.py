from __future__ import annotations

from abc import ABC, abstractmethod

from src.benchmark.schemas import BenchmarkItem
from src.retrieval.schemas import RetrievalResult


class RetrievalEvaluator(ABC):
    """
    Placeholder interface for retrieval metrics (recall@k, mrr, hit@k).
    """

    @abstractmethod
    def evaluate(self, benchmark: list[BenchmarkItem], predictions: list[RetrievalResult]) -> dict[str, float]:
        raise NotImplementedError
