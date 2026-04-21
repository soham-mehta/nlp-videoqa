from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from config import (
    PATHS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
    WHISPER_VAD_FILTER,
)
from src.utils import ensure_dir, read_json, write_json


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def extract_audio_wav(video_path: Path, wav_path: Path) -> None:
    """
    Extract mono 16kHz WAV via ffmpeg. Assumes ffmpeg is installed and on PATH.
    """
    ensure_dir(wav_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(wav_path),
    ]
    try:
        # Keep ffmpeg quiet on success; show details only on error.
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed for {video_path}: {e}") from e


def transcribe_video_id(video_id: str) -> Path:
    """
    Transcribe a downloaded video using faster-whisper and write JSON segments.
    Returns transcript JSON path.
    """
    video_md_path = PATHS.videos_metadata_dir / f"{video_id}.json"
    if not video_md_path.exists():
        raise FileNotFoundError(f"Missing video metadata: {video_md_path}")
    video_md: dict[str, Any] = read_json(video_md_path)

    video_path = Path(video_md["local_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Missing local video file: {video_path}")

    wav_path = PATHS.audio / f"{video_id}.wav"
    print(f"[transcribe] extracting audio: video_id={video_id} -> {wav_path.name}")
    extract_audio_wav(video_path, wav_path)

    print(
        "[transcribe] loading model "
        f"size={WHISPER_MODEL_SIZE} device={WHISPER_DEVICE} compute_type={WHISPER_COMPUTE_TYPE}"
    )
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    print(f"[transcribe] transcribing: video_id={video_id}")
    segments, info = model.transcribe(str(wav_path), vad_filter=WHISPER_VAD_FILTER)

    out_segments: list[dict[str, Any]] = []
    for s in segments:
        out_segments.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": (s.text or "").strip(),
            }
        )

    transcript = {
        "video_id": video_id,
        "language": getattr(info, "language", None),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "segments": out_segments,
    }

    out_path = PATHS.transcripts / f"{video_id}.json"
    write_json(out_path, transcript)
    print(f"[transcribe] wrote: {out_path}")
    return out_path


def transcribe_all_from_index() -> list[Path]:
    """
    Transcribe all videos listed in `data/metadata/videos_index.json`.
    """
    index_path = PATHS.videos_index_json
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing {index_path}. Run src/download_videos.py first."
        )
    items: list[dict[str, Any]] = read_json(index_path)
    out: list[Path] = []
    for idx, it in enumerate(items, start=1):
        vid = str(it["video_id"])
        print(f"[transcribe] ({idx}/{len(items)}) starting video_id={vid}")
        out.append(transcribe_video_id(vid))
    return out


def main() -> None:
    paths = transcribe_all_from_index()
    print(f"Wrote {len(paths)} transcripts to {PATHS.transcripts}")


if __name__ == "__main__":
    main()
