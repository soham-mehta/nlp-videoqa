"""Helper to inspect transcript segments at given timestamp ranges.
Usage: python lookup_transcript.py <video_id> <start_sec> <end_sec>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    video_id = sys.argv[1]
    start = float(sys.argv[2])
    end = float(sys.argv[3])
    path = ROOT / "hf_data" / "transcripts" / f"{video_id}.json"
    data = json.loads(path.read_text())
    for i, seg in enumerate(data["segments"]):
        if seg["end"] < start or seg["start"] > end:
            continue
        print(f"[{i:04d}] {seg['start']:7.2f}-{seg['end']:7.2f}  {seg['text']}")


if __name__ == "__main__":
    main()
