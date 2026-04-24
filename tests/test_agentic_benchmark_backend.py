import json
from pathlib import Path

import numpy as np

from src.agentic.benchmark_backend import BenchmarkRetrievalBackend
from src.retrieval.vector_store import VectorHit


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        _ = texts
        return np.array([[1.0, 0.0]], dtype=np.float32)

    def embed_images(self, images):
        _ = images
        return np.zeros((0, 0), dtype=np.float32)


class FakeVectorStore:
    def search(self, query_embedding, top_k: int, filters=None):
        _ = query_embedding
        _ = filters
        hits = [
            VectorHit(item_id="vid1:text:t1", score=0.9),
            VectorHit(item_id="vid1:frame:f1", score=0.8),
            VectorHit(item_id="vid2:text:t2", score=0.7),
        ]
        return hits[:top_k]


def _write_metadata(path: Path) -> None:
    rows = [
        {
            "item_id": "vid1:text:t1",
            "video_id": "vid1",
            "modality": "text",
            "timestamp_start": 5.0,
            "timestamp_end": 8.0,
            "source_id": "t1",
            "text": "create a virtual environment",
            "frame_path": None,
        },
        {
            "item_id": "vid1:frame:f1",
            "video_id": "vid1",
            "modality": "frame",
            "timestamp_start": 6.0,
            "timestamp_end": 7.0,
            "source_id": "f1",
            "text": None,
            "frame_path": "/tmp/frame1.jpg",
        },
        {
            "item_id": "vid2:text:t2",
            "video_id": "vid2",
            "modality": "text",
            "timestamp_start": 1.0,
            "timestamp_end": 2.0,
            "source_id": "t2",
            "text": "other video",
            "frame_path": None,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_semantic_search_respects_video_and_modality_filters(tmp_path):
    metadata_path = tmp_path / "items.jsonl"
    _write_metadata(metadata_path)
    backend = BenchmarkRetrievalBackend(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        metadata_jsonl_path=str(metadata_path),
    )

    hits = backend.semantic_search("virtual environment", k=5, video_id="vid1", modality="text")

    assert [hit.item_id for hit in hits] == ["vid1:text:t1"]


def test_get_chunks_by_timestamp_returns_overlapping_items(tmp_path):
    metadata_path = tmp_path / "items.jsonl"
    _write_metadata(metadata_path)
    backend = BenchmarkRetrievalBackend(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        metadata_jsonl_path=str(metadata_path),
    )

    hits = backend.get_chunks_by_timestamp("vid1", t_start=6.0, t_end=6.5)

    assert {hit.item_id for hit in hits} == {"vid1:text:t1", "vid1:frame:f1"}


def test_resolve_chunks_preserves_requested_order(tmp_path):
    metadata_path = tmp_path / "items.jsonl"
    _write_metadata(metadata_path)
    backend = BenchmarkRetrievalBackend(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        metadata_jsonl_path=str(metadata_path),
    )

    hits = backend.resolve_chunks(["vid1:frame:f1", "vid1:text:t1"])

    assert [hit.item_id for hit in hits] == ["vid1:frame:f1", "vid1:text:t1"]
