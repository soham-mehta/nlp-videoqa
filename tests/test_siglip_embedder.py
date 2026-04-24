import numpy as np
import torch

from src.config.settings import EmbeddingConfig
from src.models.siglip_embedder import SigLIP2EmbeddingModel


class _FakeBatch(dict):
    def to(self, device):
        _ = device
        return self


class _FakeProcessor:
    def __call__(self, **kwargs):
        _ = kwargs
        return _FakeBatch(
            {
                "input_ids": torch.ones((1, 81), dtype=torch.long),
                "attention_mask": torch.ones((1, 81), dtype=torch.long),
            }
        )


class _FakeModelConfig:
    class text_config:
        max_position_embeddings = 64


class _FakeModel:
    config = _FakeModelConfig()

    def get_text_features(self, **encoded):
        assert encoded["input_ids"].shape == (1, 64)
        assert encoded["attention_mask"].shape == (1, 64)
        return torch.ones((1, 3), dtype=torch.float32)


def test_embed_texts_clamps_overlong_processor_output():
    model = SigLIP2EmbeddingModel.__new__(SigLIP2EmbeddingModel)
    model.config = EmbeddingConfig(normalize=True)
    model.device = torch.device("cpu")
    model.processor = _FakeProcessor()
    model.model = _FakeModel()

    vectors = model.embed_texts(["x" * 500])

    assert vectors.shape == (1, 3)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)
