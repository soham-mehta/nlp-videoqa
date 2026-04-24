from __future__ import annotations

import argparse
from pathlib import Path

import modal


def _put_if_exists(batch: object, local_path: Path, remote_path: str) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Missing local asset: {local_path}")
    batch.put_file(local_path, remote_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload selected hf_data assets into a Modal Volume.")
    parser.add_argument("--volume-name", type=str, default="nlp-videoqa-assets")
    parser.add_argument("--source-root", type=Path, default=Path("hf_data"))
    parser.add_argument("--video-id", action="append", dest="video_ids", required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    volume = modal.Volume.from_name(args.volume_name, create_if_missing=True)

    with volume.batch_upload(force=True) as batch:
        for video_id in sorted(set(args.video_ids)):
            _put_if_exists(
                batch,
                source_root / "transcripts" / f"{video_id}.json",
                f"/hf_data/transcripts/{video_id}.json",
            )
            _put_if_exists(
                batch,
                source_root / "metadata" / "frames" / f"{video_id}.json",
                f"/hf_data/metadata/frames/{video_id}.json",
            )
            frames_dir = source_root / "frames" / video_id
            if not frames_dir.exists():
                raise FileNotFoundError(f"Missing frame directory: {frames_dir}")
            for frame_path in sorted(frames_dir.glob("*.jpg")):
                batch.put_file(frame_path, f"/hf_data/frames/{video_id}/{frame_path.name}")

    print(
        f"Uploaded assets for {len(set(args.video_ids))} video(s) to Modal Volume "
        f"{args.volume_name!r}: {', '.join(sorted(set(args.video_ids)))}"
    )


if __name__ == "__main__":
    main()
