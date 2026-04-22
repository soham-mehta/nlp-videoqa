from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.config.settings import EmbeddingConfig
from src.models.base import EmbeddingModel


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class SigLIP2EmbeddingModel(EmbeddingModel):
    """
    Shared-space text/image embedding model using Hugging Face SigLIP/SigLIP2 checkpoints.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
        self.model.eval()

    def _batch_indices(self, n: int) -> Iterable[tuple[int, int]]:
        bs = max(1, self.config.batch_size)
        for i in range(0, n, bs):
            yield i, min(i + bs, n)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        output: list[np.ndarray] = []
        with torch.no_grad():
            for start, end in self._batch_indices(len(texts)):
                batch = texts[start:end]
                encoded = self.processor(text=batch, padding=True, return_tensors="pt").to(self.device)
                features = self.model.get_text_features(**encoded)
                output.append(features.detach().cpu().numpy().astype(np.float32))
        vectors = np.concatenate(output, axis=0)
        return _l2_normalize(vectors) if self.config.normalize else vectors

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        output: list[np.ndarray] = []
        with torch.no_grad():
            for start, end in self._batch_indices(len(images)):
                batch = images[start:end]
                encoded = self.processor(images=batch, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**encoded)
                output.append(features.detach().cpu().numpy().astype(np.float32))
        vectors = np.concatenate(output, axis=0)
        return _l2_normalize(vectors) if self.config.normalize else vectors
