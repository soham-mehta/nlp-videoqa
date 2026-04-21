from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import PATHS
from src.utils import nearest_index, overlaps, read_json


@dataclass(frozen=True)
class LoadedVideo:
    video_id: str
    video_metadata: dict[str, Any]
    transcript: dict[str, Any] | None
    frames_metadata: dict[str, Any] | None


class VideoQADataloader:
    """
    Minimal dataloader for MVP video QA.

    Assumes the following exist:
    - `data/metadata/videos/<video_id>.json` (from downloader)
    - `data/transcripts/<video_id>.json` (from transcribe)
    - `data/metadata/frames/<video_id>.json` (from extract_frames)
    """

    def __init__(
        self,
        *,
        videos_index_path: Path | None = None,
        videos_metadata_dir: Path | None = None,
        transcripts_dir: Path | None = None,
        frames_metadata_dir: Path | None = None,
    ) -> None:
        self.videos_index_path = videos_index_path or PATHS.videos_index_json
        self.videos_metadata_dir = videos_metadata_dir or PATHS.videos_metadata_dir
        self.transcripts_dir = transcripts_dir or PATHS.transcripts
        self.frames_metadata_dir = frames_metadata_dir or PATHS.frames_metadata_dir

        items: list[dict[str, Any]] = read_json(self.videos_index_path)
        self._video_ids = [str(it["video_id"]) for it in items]

        self._frames_cache: dict[str, dict[str, Any]] = {}
        self._transcripts_cache: dict[str, dict[str, Any]] = {}

        # Precompute timestamp indices for frames.
        self._frame_timestamps: dict[str, list[float]] = {}
        self._frame_records: dict[str, list[dict[str, Any]]] = {}

        for vid in self._video_ids:
            frames_md_path = self.frames_metadata_dir / f"{vid}.json"
            if frames_md_path.exists():
                md = read_json(frames_md_path)
                frames = list(md.get("frames") or [])
                ts = [float(f["timestamp_sec"]) for f in frames]
                self._frames_cache[vid] = md
                self._frame_timestamps[vid] = ts
                self._frame_records[vid] = frames

            transcript_path = self.transcripts_dir / f"{vid}.json"
            if transcript_path.exists():
                self._transcripts_cache[vid] = read_json(transcript_path)

    def list_video_ids(self) -> list[str]:
        return list(self._video_ids)

    def load_video(self, video_id: str) -> LoadedVideo:
        md_path = self.videos_metadata_dir / f"{video_id}.json"
        if not md_path.exists():
            raise FileNotFoundError(f"Missing video metadata: {md_path}")
        video_md = read_json(md_path)

        transcript = self._transcripts_cache.get(video_id)
        frames_md = self._frames_cache.get(video_id)
        return LoadedVideo(
            video_id=video_id,
            video_metadata=video_md,
            transcript=transcript,
            frames_metadata=frames_md,
        )

    def get_frame_at_time(self, video_id: str, timestamp: float) -> dict[str, Any]:
        """
        Nearest timestamp matching for frame lookup.
        Returns a single frame record dict from frames metadata.
        """
        ts = self._frame_timestamps.get(video_id) or []
        if not ts:
            raise FileNotFoundError(
                f"No frames metadata loaded for video_id={video_id}. "
                f"Expected {self.frames_metadata_dir / f'{video_id}.json'}"
            )
        idx = nearest_index(ts, float(timestamp))
        return dict(self._frame_records[video_id][idx])

    def get_n_frames_at_time(
        self, video_id: str, timestamp: float, *, n: int = 3, step_sec: float = 5.0
    ) -> list[dict[str, Any]]:
        """
        Return n frames around the given time by sampling nearest frames at
        timestamp + k*step_sec for k in [-(n//2)..].
        """
        if n <= 0:
            return []
        center = n // 2
        out: list[dict[str, Any]] = []
        for k in range(n):
            t = float(timestamp) + (k - center) * float(step_sec)
            out.append(self.get_frame_at_time(video_id, t))
        return out

    def get_transcript_at_time(
        self, video_id: str, timestamp: float, *, window_sec: float = 10.0
    ) -> list[dict[str, Any]]:
        """
        Return all transcript segments overlapping [timestamp-window_sec/2, timestamp+window_sec/2).
        """
        tr = self._transcripts_cache.get(video_id)
        if not tr:
            transcript_path = self.transcripts_dir / f"{video_id}.json"
            raise FileNotFoundError(
                f"No transcript loaded for video_id={video_id}. Expected {transcript_path}"
            )
        segs = list(tr.get("segments") or [])
        half = float(window_sec) / 2.0
        start = float(timestamp) - half
        end = float(timestamp) + half
        out: list[dict[str, Any]] = []
        for s in segs:
            s0 = float(s["start"])
            s1 = float(s["end"])
            if overlaps(s0, s1, start, end):
                out.append(dict(s))
        return out

