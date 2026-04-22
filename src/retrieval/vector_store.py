from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorHit:
    item_id: str
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add(self, items: list[str], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path, ids_path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path, ids_path: Path) -> "VectorStore":
        raise NotImplementedError


# TODO: implement PgVectorStore with the same interface.
