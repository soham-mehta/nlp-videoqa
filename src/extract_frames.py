from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from config import FRAME_IMAGE_EXT, FRAME_INTERVAL_SEC, PATHS
from src.utils import ensure_dir, read_json, write_json


@dataclass(frozen=True)
class FrameRecord:
    timestamp_sec: float
    frame_index: int
    image_path: str
    width: int
    height: int


def extract_frames_for_video_id(
    video_id: str,
    *,
    interval_sec: float = FRAME_INTERVAL_SEC,
) -> Path:
    """
    Extract frames every `interval_sec` seconds and write frame metadata JSON.
    Frames are stored under `data/frames/<video_id>/`.
    Returns frame metadata path.
    """
    video_md_path = PATHS.videos_metadata_dir / f"{video_id}.json"
    if not video_md_path.exists():
        raise FileNotFoundError(f"Missing video metadata: {video_md_path}")
    video_md: dict[str, Any] = read_json(video_md_path)

    video_path = Path(video_md["local_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Missing local video file: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (frame_count / fps) if fps > 0 else None

    out_dir = PATHS.frames / video_id
    ensure_dir(out_dir)

    frames: list[dict[str, Any]] = []
    t = 0.0
    i = 0
    try:
        while True:
            if duration_sec is not None and t > duration_sec:
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            h, w = frame.shape[:2]
            image_path = out_dir / f"{int(round(t * 1000.0)):010d}{FRAME_IMAGE_EXT}"
            ok2 = cv2.imwrite(str(image_path), frame)
            if not ok2:
                raise RuntimeError(f"Failed to write frame image: {image_path}")

            frames.append(
                {
                    "timestamp_sec": float(t),
                    "frame_index": int(i),
                    "image_path": str(image_path),
                    "width": int(w),
                    "height": int(h),
                }
            )
            i += 1
            t += float(interval_sec)
    finally:
        cap.release()

    md = {
        "video_id": video_id,
        "source_video_path": str(video_path),
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "duration_sec": float(duration_sec) if duration_sec is not None else None,
        "interval_sec": float(interval_sec),
        "frames": frames,
    }

    out_md_path = PATHS.frames_metadata_dir / f"{video_id}.json"
    write_json(out_md_path, md)
    return out_md_path


def extract_all_from_index() -> list[Path]:
    index_path = PATHS.videos_index_json
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing {index_path}. Run src/download_videos.py first."
        )
    items: list[dict[str, Any]] = read_json(index_path)
    out: list[Path] = []
    for it in items:
        out.append(extract_frames_for_video_id(str(it["video_id"])))
    return out


def main() -> None:
    paths = extract_all_from_index()
    print(f"Wrote {len(paths)} frame-metadata JSON files to {PATHS.frames_metadata_dir}")


if __name__ == "__main__":
    main()
