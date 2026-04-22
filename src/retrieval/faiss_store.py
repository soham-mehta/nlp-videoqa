from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.retrieval.vector_store import VectorHit, VectorStore
from src.utils.io import read_json, write_json


def _ensure_2d(embedding: np.ndarray) -> np.ndarray:
    if embedding.ndim == 1:
        return embedding.reshape(1, -1).astype(np.float32)
    return embedding.astype(np.float32)


class FaissVectorStore(VectorStore):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []

    def add(self, items: list[str], embeddings: np.ndarray) -> None:
        vectors = _ensure_2d(embeddings)
        if len(items) != vectors.shape[0]:
            raise ValueError("items and embeddings size mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected embedding dim={self.dim}, got {vectors.shape[1]}")
        self.index.add(vectors)
        self._ids.extend(items)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        # Note: FAISS cannot apply metadata filters directly here.
        # Retrieval service applies filters after joining metadata.
        _ = filters
        q = _ensure_2d(query_embedding)
        scores, idx = self.index.search(q, max(1, top_k))
        hits: list[VectorHit] = []
        for score, row_idx in zip(scores[0].tolist(), idx[0].tolist()):
            if row_idx < 0 or row_idx >= len(self._ids):
                continue
            hits.append(VectorHit(item_id=self._ids[row_idx], score=float(score)))
        return hits

    def save(self, path: Path, ids_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        write_json(ids_path, {"ids": self._ids, "dim": self.dim})

    @classmethod
    def load(cls, path: Path, ids_path: Path) -> "FaissVectorStore":
        info = read_json(ids_path)
        store = cls(dim=int(info["dim"]))
        store.index = faiss.read_index(str(path))
        store._ids = [str(x) for x in info["ids"]]
        return store
