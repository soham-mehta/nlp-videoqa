from __future__ import annotations

import base64
import os
import re
from pathlib import Path

_FRAME_ROOTS: list[str] = []


def set_frame_roots(roots: list[str]) -> None:
    """Configure fallback directories for resolving frame paths from foreign indexes."""
    _FRAME_ROOTS.clear()
    _FRAME_ROOTS.extend(roots)


def _resolve_frame_path(frame_path: str) -> Path:
    """Resolve a frame path, trying fallback roots if the original doesn't exist."""
    p = Path(frame_path)
    if p.exists():
        return p
    # Extract the relative part: video_id/frame_file.jpg
    m = re.search(r"frames/(.+)$", frame_path)
    if not m:
        return p
    rel = m.group(1)
    for root in _FRAME_ROOTS:
        candidate = Path(root) / rel
        if candidate.exists():
            return candidate
    return p


def encode_frame_base64(frame_path: str) -> str | None:
    """Load a frame file and return an OpenAI-compatible base64 data URI."""
    p = _resolve_frame_path(frame_path)
    if not p.exists():
        return None
    try:
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        suffix = p.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def build_image_content_parts(
    frame_paths: list[str],
    detail: str = "low",
) -> list[dict]:
    """Encode frames as OpenAI image_url content parts. Skips missing files."""
    parts: list[dict] = []
    for path in frame_paths:
        uri = encode_frame_base64(path)
        if uri:
            parts.append({"type": "image_url", "image_url": {"url": uri, "detail": detail}})
    return parts
