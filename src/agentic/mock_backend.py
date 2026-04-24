from __future__ import annotations

from dataclasses import replace

from src.agentic.backend import Chunk, Modality

_FIXTURES: list[Chunk] = [
    Chunk("c1", "vid1", "text", 0.0, 5.0, text="Welcome to the tutorial on setting up the API."),
    Chunk("c2", "vid1", "text", 5.0, 10.0, text="First, you need to create an API key in the dashboard."),
    Chunk("c3", "vid1", "frame", 6.0, 7.0, text="Screen shows the OpenAI dashboard with a 'Create new secret key' button."),
    Chunk("c4", "vid1", "text", 6.0, 7.0, text="sk-proj-XXXX...  Create new secret key"),
    Chunk("c5", "vid1", "text", 10.0, 15.0, text="Now let's run the tests to verify the setup."),
    Chunk("c6", "vid1", "frame", 12.0, 13.0, text="Terminal showing 3 failing tests in red."),
    Chunk("c7", "vid1", "text", 15.0, 20.0, text="The tests failed, so we need to debug the connection."),
    Chunk("c8", "vid1", "frame", 17.0, 18.0, text="Editor opens config.py; developer edits the base URL."),
]


class MockBackend:
    """In-memory stub so we can develop the agent loop before the real backend is used."""

    def semantic_search(
        self,
        query: str,
        k: int = 8,
        modality: Modality | None = None,
        video_id: str | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> list[Chunk]:
        terms = [w for w in query.lower().split() if len(w) > 2]
        scored: list[Chunk] = []
        for c in _FIXTURES:
            if modality is not None and c.modality != modality:
                continue
            if video_id is not None and c.video_id != video_id:
                continue
            if t_start is not None and c.timestamp_end < t_start:
                continue
            if t_end is not None and c.timestamp_start > t_end:
                continue
            haystack = (c.text or "").lower()
            score = sum(1 for w in terms if w in haystack)
            scored.append(replace(c, score=float(score)))
        scored.sort(key=lambda c: c.score or 0.0, reverse=True)
        return scored[:k]

    def get_chunks_by_timestamp(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        modality: Modality | None = None,
    ) -> list[Chunk]:
        return [
            c for c in _FIXTURES
            if c.video_id == video_id
            and c.timestamp_end >= t_start and c.timestamp_start <= t_end
            and (modality is None or c.modality == modality)
        ]

    def get_nearby_chunks(self, chunk_id: str, radius_seconds: float = 10.0) -> list[Chunk]:
        anchor = next((c for c in _FIXTURES if c.item_id == chunk_id), None)
        if anchor is None:
            return []
        lo, hi = anchor.timestamp_start - radius_seconds, anchor.timestamp_end + radius_seconds
        return [
            c for c in _FIXTURES
            if c.video_id == anchor.video_id and c.timestamp_end >= lo and c.timestamp_start <= hi
        ]

    def get_video_metadata(self, video_id: str) -> dict:
        if video_id != "vid1":
            return {}
        return {
            "video_id": "vid1",
            "title": "API Setup Tutorial (mock)",
            "duration_sec": 20.0,
            "modalities": ["text", "frame"],
        }
