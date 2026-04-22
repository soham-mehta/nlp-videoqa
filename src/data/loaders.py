from __future__ import annotations

from pathlib import Path

from src.config.settings import PathsConfig
from src.data.schemas import FrameItem, IndexedItem, TranscriptChunk
from src.utils.io import read_json


def discover_video_ids(paths: PathsConfig) -> list[str]:
    """
    Discover video ids from transcript JSON files.
    """
    if not paths.transcripts_dir.exists():
        return []
    return sorted([p.stem for p in paths.transcripts_dir.glob("*.json")])


def load_transcript_chunks_for_video(paths: PathsConfig, video_id: str) -> list[TranscriptChunk]:
    """
    Load sentence-level transcript chunks from one JSON file.
    Assumptions:
    - File is `data/transcripts/<video_id>.json`
    - Segment records contain text and start/end-like fields.
    """
    transcript_path = paths.transcripts_dir / f"{video_id}.json"
    if not transcript_path.exists():
        return []
    payload = read_json(transcript_path)
    segments = payload.get("segments", [])
    chunks: list[TranscriptChunk] = []
    for idx, seg in enumerate(segments):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", seg.get("timestamp_start", 0.0)) or 0.0)
        end = float(seg.get("end", seg.get("timestamp_end", start)) or start)
        source_id = str(seg.get("chunk_id", f"seg_{idx:06d}"))
        chunks.append(
            TranscriptChunk(
                video_id=video_id,
                source_id=source_id,
                timestamp_start=start,
                timestamp_end=end,
                text=text,
            )
        )
    return chunks


def enumerate_frames_for_video(paths: PathsConfig, video_id: str) -> list[FrameItem]:
    """
    Enumerate frame records for one video.
    Prefers `data/metadata/frames/<video_id>.json`, falls back to image file glob.
    """
    metadata_path = paths.frames_metadata_dir / f"{video_id}.json"
    if metadata_path.exists():
        payload = read_json(metadata_path)
        out: list[FrameItem] = []
        for idx, frame in enumerate(payload.get("frames", [])):
            ts = float(frame.get("timestamp_sec", idx))
            image_path = str(frame.get("image_path", ""))
            if not image_path:
                continue
            out.append(
                FrameItem(
                    video_id=video_id,
                    source_id=str(frame.get("frame_index", idx)),
                    timestamp_start=ts,
                    timestamp_end=ts + 1.0,
                    frame_path=image_path,
                )
            )
        return out

    frame_dir = paths.frames_dir / video_id
    if not frame_dir.exists():
        return []
    files = sorted(frame_dir.glob("*.jpg"))
    fallback: list[FrameItem] = []
    for idx, file in enumerate(files):
        # Current pipeline uses millisecond filenames like 0000001000.jpg.
        try:
            ts = float(int(file.stem) / 1000.0)
        except ValueError:
            ts = float(idx)
        fallback.append(
            FrameItem(
                video_id=video_id,
                source_id=file.stem,
                timestamp_start=ts,
                timestamp_end=ts + 1.0,
                frame_path=str(file),
            )
        )
    return fallback


def sample_frames_at_fps(frames: list[FrameItem], target_fps: float = 1.0) -> list[FrameItem]:
    """
    Sample frames close to regular wall-clock bins.
    For 1 FPS, keeps at most one frame per integer second.
    """
    if not frames:
        return []
    if target_fps <= 0:
        return list(frames)
    interval_sec = 1.0 / target_fps
    chosen: list[FrameItem] = []
    seen_bins: set[int] = set()
    for frame in sorted(frames, key=lambda x: x.timestamp_start):
        sec_bin = int(frame.timestamp_start / interval_sec)
        if sec_bin in seen_bins:
            continue
        seen_bins.add(sec_bin)
        chosen.append(frame)
    return chosen


def build_indexed_items_from_video(paths: PathsConfig, video_id: str, frame_fps: float) -> list[IndexedItem]:
    chunks = load_transcript_chunks_for_video(paths, video_id)
    frames = sample_frames_at_fps(enumerate_frames_for_video(paths, video_id), target_fps=frame_fps)
    items: list[IndexedItem] = []
    for chunk in chunks:
        items.append(
            IndexedItem(
                item_id=f"{video_id}:text:{chunk.source_id}",
                video_id=video_id,
                modality="text",
                timestamp_start=chunk.timestamp_start,
                timestamp_end=chunk.timestamp_end,
                source_id=chunk.source_id,
                text=chunk.text,
            )
        )
    for frame in frames:
        items.append(
            IndexedItem(
                item_id=f"{video_id}:frame:{frame.source_id}",
                video_id=video_id,
                modality="frame",
                timestamp_start=frame.timestamp_start,
                timestamp_end=frame.timestamp_end,
                source_id=frame.source_id,
                frame_path=frame.frame_path,
            )
        )
    return items
