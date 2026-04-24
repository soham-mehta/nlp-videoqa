from __future__ import annotations

import json
from dataclasses import asdict

from src.agentic.backend import Chunk, RetrievalBackend

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Search indexed video chunks by semantic similarity. Returns top-k matching "
                "text and frame evidence items. Use modality or time filters to narrow the "
                "search when you already have a rough anchor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query."},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                    "modality": {
                        "type": "string",
                        "enum": ["text", "frame"],
                        "description": "Restrict to one modality.",
                    },
                    "video_id": {"type": "string"},
                    "t_start": {"type": "number", "description": "Earliest timestamp in seconds."},
                    "t_end": {"type": "number", "description": "Latest timestamp in seconds."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunks_by_timestamp",
            "description": (
                "Return all chunks overlapping a time window in one video. Useful after locating "
                "an event and needing every surrounding stream at that moment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "modality": {"type": "string", "enum": ["text", "frame"]},
                },
                "required": ["video_id", "t_start", "t_end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearby_chunks",
            "description": (
                "Given a chunk id, return chunks within a time radius in the same video. Good for "
                "expanding context around a hit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "radius_seconds": {"type": "number", "default": 10.0},
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_metadata",
            "description": "Get video metadata such as duration and available indexed evidence.",
            "parameters": {
                "type": "object",
                "properties": {"video_id": {"type": "string"}},
                "required": ["video_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Submit the final answer. Call this only once you have enough evidence. The loop "
                "terminates on this call. List the chunk ids you actually used as evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer", "evidence_chunk_ids"],
            },
        },
    },
]


def _dump_chunks(chunks: list[Chunk]) -> str:
    return json.dumps([asdict(c) for c in chunks], ensure_ascii=False)


def dispatch(backend: RetrievalBackend, name: str, args: dict) -> str:
    if name == "semantic_search":
        return _dump_chunks(backend.semantic_search(**args))
    if name == "get_chunks_by_timestamp":
        return _dump_chunks(backend.get_chunks_by_timestamp(**args))
    if name == "get_nearby_chunks":
        return _dump_chunks(backend.get_nearby_chunks(**args))
    if name == "get_video_metadata":
        return json.dumps(backend.get_video_metadata(**args))
    raise ValueError(f"unknown tool: {name}")
