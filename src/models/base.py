from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError
