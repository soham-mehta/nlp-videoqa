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


def _features_to_numpy(features: torch.Tensor | object) -> np.ndarray:
    """
    Handle both tensor outputs and model output objects.
    """
    if isinstance(features, torch.Tensor):
        tensor = features
    elif hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
        tensor = features.pooler_output
    elif hasattr(features, "last_hidden_state") and isinstance(features.last_hidden_state, torch.Tensor):
        tensor = features.last_hidden_state.mean(dim=1)
    else:
        raise TypeError(f"Unsupported feature output type: {type(features)}")
    return tensor.detach().cpu().numpy().astype(np.float32)


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

    def _max_text_length(self) -> int:
        cfg = getattr(self.model.config, "text_config", None)
        if cfg is not None and getattr(cfg, "max_position_embeddings", None):
            return int(cfg.max_position_embeddings)
        return 64

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        max_len = self._max_text_length()
        output: list[np.ndarray] = []
        with torch.no_grad():
            for start, end in self._batch_indices(len(texts)):
                batch = texts[start:end]
                encoded = self.processor(
                    text=batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                # Some processor/tokenizer combinations still return overlong tensors.
                # Clamp them defensively so retrieval queries never exceed the model limit.
                if "input_ids" in encoded and encoded["input_ids"].shape[1] > max_len:
                    for key, value in list(encoded.items()):
                        if isinstance(value, torch.Tensor) and value.ndim >= 2:
                            encoded[key] = value[:, :max_len]
                encoded = encoded.to(self.device)
                features = self.model.get_text_features(**encoded)
                output.append(_features_to_numpy(features))
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
                output.append(_features_to_numpy(features))
        vectors = np.concatenate(output, axis=0)
        return _l2_normalize(vectors) if self.config.normalize else vectors
